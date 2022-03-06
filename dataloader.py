import torch
from torch.utils.data import Dataset, DataLoader

import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2

from glob import glob
import cv2

class DataFrame(Dataset):
    def __init__(self, is_train = True, transform = None):
        self.transform = transform

        root_path = './DATA'
        data_path = (root_path + '/train') if is_train else (root_path + '/test')
        self.data_folder_list = sorted(glob(data_path + '/*'), key = lambda x : int(x.split('/')[-1]))
        self.label_data = pd.read_csv(data_path + '_label.csv')
        self.data = []

        id_to_label = self._label_function()
        for i, data_folder in enumerate(self.data_folder_list):
            image_paths = self._get_metadata(data_folder)
            label = id_to_label[self.label_data['id'][i]]
            self.data += [
                (image_path, torch.LongTensor([label])) for image_path in image_paths
            ]

    @staticmethod
    def _get_metadata(data_folder):
        image_paths = sorted(glob(data_folder + '/*.png'), key=lambda x: int(x.split('/')[-1].replace('.png', '')))
        return image_paths

    @staticmethod
    def _label_function():
        hand_gesture = pd.read_csv('../DATA/hand_gesture_pose.csv')
        label_to_id = dict(hand_gesture['pose_id'])
        id_to_label = {}
        for label, id in label_to_id.items():
            id_to_label[id] = label
        return id_to_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        image_path, label = self.data[item]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        return image, label.type(torch.LongTensor)

def Dataloader(is_train, batch_size, num_workers=0):

    if is_train:
        train_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=224),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
                A.RandomCrop(height=224, width=224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        dataset = DataFrame(is_train, train_transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True,
        )
    else:
        test_transform = A.Compose(
            [
                A.SmallestMaxSize(max_size=224),
                A.CenterCrop(height=224, width=224),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ]
        )
        dataset = DataFrame(is_train, test_transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True,
        )
    return dataloader