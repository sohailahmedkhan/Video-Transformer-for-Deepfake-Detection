import numpy as np
import glob, os
from .augmentations import *
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

class DeepFakeSet(Dataset):
    def __init__(self, file_list, transform=None):

        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        self.filelength = len(self.file_list)
        return self.filelength

    def __getitem__(self, idx):
        img_path = self.file_list[idx]
        img = Image.open(img_path)
        img_transformed = self.transform(img)

        label = img_path.split('\\')[-2].split('.')[0].split('/')[-1]
        label = 1 if label == "real" else 0

        return img_transformed, label

class TrainDataset:
    def get_image_batches(paths, batch_size):

        batch_size = batch_size

        train_list_real = glob.glob(os.path.join(paths[0],'*'))
        train_list_fake = glob.glob(os.path.join(paths[1],'*'))
        train_list_fake_2 = glob.glob(os.path.join(paths[2],'*'))
        train_list_fake_3 = glob.glob(os.path.join(paths[3],'*'))
        train_list_fake_4 = glob.glob(os.path.join(paths[4],'*'))


        valid_list_real = glob.glob(os.path.join(paths[5],'*'))
        valid_list_fake = glob.glob(os.path.join(paths[6],'*'))
        valid_list_fake_2 = glob.glob(os.path.join(paths[7],'*'))
        valid_list_fake_3 = glob.glob(os.path.join(paths[8],'*'))
        valid_list_fake_4 = glob.glob(os.path.join(paths[9],'*'))


        train_list = []
        train_list.extend(train_list_real[:100000])
        train_list.extend(train_list_fake[:25000])
        train_list.extend(train_list_fake_2[:25000])
        train_list.extend(train_list_fake_3[:25000])
        train_list.extend(train_list_fake_4[:25000])

        valid_list = []
        valid_list.extend(valid_list_real[:20000])
        valid_list.extend(valid_list_fake[:5000])
        valid_list.extend(valid_list_fake_2[:5000])
        valid_list.extend(valid_list_fake_3[:5000])
        valid_list.extend(valid_list_fake_4[:5000])

        print(f"Train Data Real: {len(train_list_real)}")
        print(f"Train Data Fake: {len(train_list_fake)}")
        print(f"Train Data Fake 2: {len(train_list_fake_2)}")
        print(f"Train Data Fake 3: {len(train_list_fake_3)}")
        print(f"Train Data Fake 4: {len(train_list_fake_4)}")


        np.random.shuffle(train_list)
        np.random.shuffle(train_list)
        np.random.shuffle(train_list)
        np.random.shuffle(train_list)

        np.random.shuffle(valid_list)
        np.random.shuffle(valid_list)
        np.random.shuffle(valid_list)
        np.random.shuffle(valid_list)

        labels = [path.split('\\')[-2].split('.')[0].split('/')[-1] for path in train_list]
        print("Labels: " + str(labels[:10]))

        train_data = DeepFakeSet(train_list, transform=transforms_imgaug)
        valid_data = DeepFakeSet(valid_list, transform=val_transforms)

        train_loader = DataLoader(dataset = train_data, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(dataset = valid_data, batch_size=batch_size, shuffle=True)


        print(len(train_data), len(train_loader))
        print(len(valid_data), len(valid_loader))

        return train_loader, valid_loader