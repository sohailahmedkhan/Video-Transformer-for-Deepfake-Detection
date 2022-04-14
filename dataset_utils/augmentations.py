from imgaug import augmenters as iaa
import torchvision.transforms as transforms
import numpy as np
import torchvision


class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
        iaa.Resize((299, 299)),
#         iaa.Crop(px=(0, 16)),
        iaa.Fliplr(0.5), # horizontally flip
        iaa.OneOf([
            iaa.Affine(scale=2),
            iaa.Affine(rotate=20),
            iaa.Affine(translate_px=(-20, 20)),
            iaa.Cutout(fill_mode="constant", cval=0, nb_iterations=2, size=0.4)
        ]),
#         iaa.OneOf([
# #             iaa.JpegCompression(compression=(60, 99)),
# #             iaa.GaussianBlur((0, 2.0)),
# #             iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
# #             iaa.Multiply((0.5, 2.0), per_channel=0.2),
#             iaa.Cutout(fill_mode="constant", cval=0, nb_iterations=1, size=0.5)
#                 ])
        ], random_order=True)
      
    def __call__(self, img):
        img = np.array(img).astype(np.uint8)
        img = self.aug.augment_image(img)
        img = torchvision.transforms.ToTensor()(img)
        # img = torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(img)
        return img
transforms_imgaug = ImgAugTransform()

# class ImgAugTransform:
#     def __init__(self):
#         self.aug = iaa.Sequential([
#         iaa.Resize((299, 299)),
#         iaa.OneOf([
#             iaa.Cutout(fill_mode="constant", cval=0, nb_iterations=2, size=0.3)
#         ])], random_order=True)
      
#     def __call__(self, img):
#         img = np.array(img).astype(np.uint8)
#         img = self.aug.augment_image(img)
#         img = torchvision.transforms.ToTensor()(img)
# #         img = torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
#         return img
# transforms_imgaug = ImgAugTransform()

train_transforms = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


test_transforms = transforms.Compose(
    [
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ]
)