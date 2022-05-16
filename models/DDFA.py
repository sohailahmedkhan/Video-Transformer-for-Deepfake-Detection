import os.path as osp
import scipy.io as sio

import torch
import numpy as np
from torchvision import datasets, transforms
from Sim3DR import rasterize
from utils.functions import plot_image
from utils.io import _load
import yaml
from utils.tddfa_util import _to_ctype
class DDFA():

    def load_uv_coords(fp):
        C = sio.loadmat(fp)
        uv_coords = C['UV'].copy(order='C').astype(np.float32)
        return uv_coords


    def process_uv(uv_coords, uv_h=256, uv_w=256):
        uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
        uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
        uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
        uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1), dtype=np.float32)))  # add z
        return uv_coords


    def get_colors(img, ver):
        # nearest-neighbor sampling
        [h, w, _] = img.shape
        ver[0, :] = np.minimum(np.maximum(ver[0, :], 0), w - 1)  # x
        ver[1, :] = np.minimum(np.maximum(ver[1, :], 0), h - 1)  # y
        ind = np.round(ver).astype(np.int32)
        colors = img[ind[1, :], ind[0, :], :]  # n x 3

        return colors


    def bilinear_interpolate(img, x, y):
        """
        https://stackoverflow.com/questions/12729228/simple-efficient-bilinear-interpolation-of-images-in-numpy-and-python
        """
        x0 = np.floor(x).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(y).astype(np.int32)
        y1 = y0 + 1

        x0 = np.clip(x0, 0, img.shape[1] - 1)
        x1 = np.clip(x1, 0, img.shape[1] - 1)
        y0 = np.clip(y0, 0, img.shape[0] - 1)
        y1 = np.clip(y1, 0, img.shape[0] - 1)

        i_a = img[y0, x0]
        i_b = img[y1, x0]
        i_c = img[y0, x1]
        i_d = img[y1, x1]

        wa = (x1 - x) * (y1 - y)
        wb = (x1 - x) * (y - y0)
        wc = (x - x0) * (y1 - y)
        wd = (x - x0) * (y - y0)

        return wa[..., np.newaxis] * i_a + wb[..., np.newaxis] * i_b + wc[..., np.newaxis] * i_c + wd[..., np.newaxis] * i_d


    def uv_tex(img, ver_lst, tri, g_uv_coords,uv_h=299, uv_w=299, uv_c=3, show_flag=False, wfp=None):
        uv_coords = DDFA.process_uv(g_uv_coords, uv_h=uv_h, uv_w=uv_w)

        res_lst = []
        for ver_ in ver_lst:
            ver = _to_ctype(ver_.T)  # transpose to m x 3
            colors = DDFA.bilinear_interpolate(img, ver[:, 0], ver[:, 1]) / 255.
            # `rasterize` here serves as texture sampling, may need to optimization
            res = rasterize(uv_coords, tri, colors, height=uv_h, width=uv_w, channel=uv_c)
            res_lst.append(res)

        # concat if there more than one image
        res = np.concatenate(res_lst, axis=1) if len(res_lst) > 1 else res_lst[0]

        if wfp is not None:
            cv2.imwrite(wfp, res)
            print(f'Save visualization result to {wfp}')

        if show_flag:
            plot_image(res)

        return res

    def load_configs():
        # load config
        cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
        # Init FaceBoxes and TDDFA, recommend using onnx flag
        onnx_flag = True  # or True to use ONNX to speed up
        if onnx_flag:
            import os
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
            os.environ['OMP_NUM_THREADS'] = '4'

            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from TDDFA_ONNX import TDDFA_ONNX

            face_boxes = FaceBoxes_ONNX()
            tddfa = TDDFA_ONNX(**cfg)
        else:
            tddfa = TDDFA(gpu_mode=False, **cfg)
            face_boxes = FaceBoxes()
        return tddfa, face_boxes
    
    def generate_uv_tex(x, features, sequence_embedding, linear_1, linear_3, proj, seq_embed, hybrid, variant, device):
        tddfa, face_boxes = DDFA.load_configs()
        to_pil_image = transforms.ToPILImage()
        to_tensor = transforms.ToTensor()
        to_normalized = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        if variant == 'image':
            b, c, fh, fw = x.shape
            all_concated_feats = torch.zeros(1, 324, 768).to(device)
        else:
            b = 1
            all_concated_feats = torch.zeros(1, 64, 768).to(device)
        
        for i in range(b):
            image = x[i].unsqueeze(0)
            image = to_normalized(image)
            
            if hybrid == True:
                image_features = features(image)
                image_features = image_features[-1]
                image_features = proj(image_features).flatten(2)
                image_features = linear_1(image_features)
                image_features = image_features.transpose(2,1)

            elif variant == 'image' and hybrid == False:
                image_features = features(image)
                image_features = image_features.flatten(2).transpose(1, 2)

            elif variant == 'video' and hybrid == False:
                image_features = features(image)
                image_features = image_features.flatten(2)
                image_features = linear_3(image_features)
                image_features = image_features.transpose(2,1)

            if variant=='image' and seq_embed==True:
                image_features = sequence_embedding(image_features)
    
            img = to_pil_image(x[i])
            img = np.asarray(img)
            boxes = face_boxes(img)
            
            if len(boxes) == 0 or len(boxes)>1:
                if hybrid == True:
                    uv_features = features(torch.zeros_like(image).to(device))
                    uv_features = uv_features[-1]
                    uv_features = proj(uv_features).flatten(2)
                    uv_features = linear_1(uv_features)
                    uv_features = uv_features.transpose(2,1)

                elif variant == 'image' and hybrid == False:
                    uv_features = features(torch.zeros_like(image).to(device))
                    uv_features = uv_features.flatten(2).transpose(1, 2)

                elif variant == 'video' and hybrid == False:
                    uv_features = features(torch.zeros_like(image).to(device))
                    uv_features = uv_features.flatten(2)
                    uv_features = linear_3(uv_features)
                    uv_features = uv_features.transpose(2,1)

                if variant=='image' and seq_embed==True:
                    uv_features = sequence_embedding(uv_features)

                uv_features = torch.cat((image_features, uv_features), dim=1)

            else:
                param_lst, roi_box_lst = tddfa(img, boxes)
                g_uv_coords = DDFA.load_uv_coords(('configs/BFM_UV.mat'))
                indices = _load(('configs/indices.npy'))  # todo: handle bfm_slim
                g_uv_coords = g_uv_coords[indices, :]
                ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
                uv_tensor = to_tensor(DDFA.uv_tex(img, ver_lst, tddfa.tri, g_uv_coords, show_flag=False, wfp=None))
                uv_tensor = uv_tensor.unsqueeze(0)
                uv_tensor = to_normalized(uv_tensor)
                if hybrid == True:
                    uv_features = features(uv_tensor.to(device))
                    uv_features = uv_features[-1]
                    uv_features = proj(uv_features).flatten(2)
                    uv_features = linear_1(uv_features)
                    uv_features = uv_features.transpose(2,1)

                elif variant == 'image' and hybrid == False:
                    uv_features = features(uv_tensor.to(device))
                    uv_features = uv_features.flatten(2).transpose(1, 2)

                elif variant == 'video' and hybrid == False:
                    uv_features = features(uv_tensor.to(device))
                    uv_features = uv_features.flatten(2)
                    uv_features = linear_3(uv_features)
                    uv_features = uv_features.transpose(2,1)

                if variant=='image' and seq_embed==True:
                    uv_features = sequence_embedding(uv_features)

                uv_features = torch.cat((image_features, uv_features), dim=1)
            all_concated_feats = torch.cat((all_concated_feats, uv_features), dim=0)
        return all_concated_feats[1:]