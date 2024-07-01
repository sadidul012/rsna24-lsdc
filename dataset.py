import os
import random

import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import torch
import tqdm
from torch.utils.data import Dataset
import albumentations as A
import pydicom
import pickle
import cv2


def convert_to_8bit(x):
    lower, upper = np.percentile(x, (1, 99))
    x = np.clip(x, lower, upper)
    x = x - np.min(x)
    x = x / np.max(x)
    return (x * 255).astype("uint8")


def load_dicom_stack(dicom_folder, plane, reverse_sort=False):
    dicom_files = glob(os.path.join(dicom_folder, "*.dcm"))
    dicoms = [pydicom.dcmread(f) for f in dicom_files]
    plane = {"sagittal": 0, "coronal": 1, "axial": 2}[plane.lower()]
    positions = np.asarray([float(d.ImagePositionPatient[plane]) for d in dicoms])
    idx = np.argsort(-positions if reverse_sort else positions)
    ipp = np.asarray([d.ImagePositionPatient for d in dicoms]).astype("float")[idx]
    try:
        array = []
        for d in dicoms:
            d = d.pixel_array.astype("float32")
            d = cv2.resize(d, (512, 512))
            array.append(d)
        array = np.stack(array)
        array = array[idx]
        return {
            "array": convert_to_8bit(array),
            "positions": ipp,
            "pixel_spacing": np.asarray(dicoms[0].PixelSpacing).astype("float")
        }
    except Exception as e:
        print(e)
        return {
            "array": convert_to_8bit(np.zeros((2, 200, 200), np.float32)),
            "positions": np.zeros((2, 3), np.float32),
            "pixel_spacing": np.zeros(2).astype("float")
        }


class RSNA24DatasetValid(Dataset):
    def __init__(self, df, train_des, image_dir, plane=None, in_channel=30, image_size=(512, 512), split="valid", use_cache=True):
        self.df = df
        self.train_des = train_des
        self.image_dir = image_dir
        self.image_size = image_size
        self.in_channel = in_channel
        self.split = split
        self.use_cache = use_cache
        self.planes = {"Sagittal T2/STIR": "sagittal", "Sagittal T1": "sagittal", "Axial T2": "axial"}
        self.plane_names = list(self.planes.keys())
        self.label_indexes = {
            "Sagittal T2/STIR": [0, 5],
            "Sagittal T1": [5, 15],
            "Axial T2": [15, 25]
        }
        self.plane = plane

        self.df = self.df.fillna(-100)
        self.label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
        self.df = self.df.replace(self.label2id)

        self.transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=0.5, std=0.5)
        ])

    def __len__(self):
        return len(self.df)

    def process(self, data, return_length):
        if data is None:
            return np.zeros((return_length, self.image_size[0], self.image_size[1]))

        step = len(data["array"]) / return_length
        st = 0
        end = len(data["array"])
        images = []
        positions = []
        for j, i in enumerate(list(np.arange(st, end, step))):
            ind = max(0, int((i - 0.5001).round()))
            images.append(data["array"][ind])
            positions.append(data["positions"][ind])

        images = np.array(images)[:return_length]
        images = images.transpose(2, 1, 0)
        if self.transform is not None:
            images = self.transform(image=images)['image']
        images = images.transpose(2, 1, 0)

        if images.shape[0] - return_length < 0:
            images = np.pad(
                images, ((abs(images.shape[0] - return_length), 0), (0, 0), (0, 0)), mode='constant', constant_values=0
            )
        return images

    def save_instance(self, value, idx):
        path = f"/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification/temp/{idx}.bz2"
        if self.use_cache:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(value, f)

    def load_instance(self, idx):
        path = f"/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification/temp/{idx}.bz2"
        if self.use_cache:
            if os.path.exists(path):
                with open(path, "rb") as file:
                    return pickle.load(file)

        return None

    def read_data(self, study, idx):
        data = self.load_instance(idx)
        if data is not None:
            return data

        data_ = dict(zip(self.planes.keys(), [None for _ in self.planes.keys()]))
        for row in study.itertuples():
            d = load_dicom_stack(
                os.path.join(self.image_dir, str(row.study_id), str(row.series_id)),
                plane=self.planes[row.series_description]
            )
            if data_[row.series_description] is None:
                data_[row.series_description] = d
            else:
                data_[row.series_description] = {
                    "array": np.concatenate((data_[row.series_description]["array"], d["array"])),
                    "positions": np.concatenate((data_[row.series_description]["positions"], d["positions"])),
                    "pixel_spacing": np.concatenate(
                        (data_[row.series_description]["pixel_spacing"], d["pixel_spacing"])),
                }

        self.save_instance(data_, idx)
        return data_

    def get_plane(self):
        return self.plane

    def __getitem__(self, idx):
        t = self.df.iloc[idx]
        self.plane = self.get_plane()
        y = t[1:].values.astype(np.int64)
        label = np.zeros(25, dtype=np.int64)
        label[self.label_indexes[self.plane][0]:self.label_indexes[self.plane][1]] = y[self.label_indexes[self.plane][0]:self.label_indexes[self.plane][1]]
        st_id = int(t['study_id'])
        study = self.train_des.loc[self.train_des.study_id == st_id]

        data_ = self.read_data(study, idx)
        feat = self.process(data_[self.plane], self.in_channel)
        return feat, label


class RSNA24DatasetTrain(RSNA24DatasetValid):
    def __init__(self, df, train_des, image_dir, plane=None, in_channel=30, aug_prob=0.75, image_size=(512, 512), use_cache=True):
        super().__init__(df, train_des, image_dir, plane=plane, in_channel=in_channel, image_size=image_size, split="train", use_cache=use_cache)
        self.transform = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=aug_prob),
            A.OneOf([
                A.MotionBlur(blur_limit=5),
                A.MedianBlur(blur_limit=5),
                A.GaussianBlur(blur_limit=5),
                A.GaussNoise(var_limit=(5.0, 30.0)),
            ], p=aug_prob),

            A.OneOf([
                A.OpticalDistortion(distort_limit=1.0),
                A.GridDistortion(num_steps=5, distort_limit=1.),
                A.ElasticTransform(alpha=3),
            ], p=aug_prob),

            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.01, border_mode=0, p=aug_prob),
            A.Resize(self.image_size[0], self.image_size[1]),
            # A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8, p=aug_prob),
            A.Normalize(mean=0.5, std=0.5)
        ])

    def get_plane(self):
        return random.choice(self.plane_names)


def main():
    # from PIL import Image
    rd = '/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification'
    df = pd.read_csv(f'{rd}/train.csv')
    df = df.iloc[:5]
    train_des = pd.read_csv(f'{rd}/train_series_descriptions.csv')
    image_dir = f"{rd}/train_images/"
    use_cache = True
    {"Sagittal T2/STIR": "sagittal", "Sagittal T1": "sagittal", "Axial T2": "axial"}
    sagittal_t2 = RSNA24DatasetValid(df, train_des, image_dir, plane="Sagittal T2/STIR", use_cache=use_cache, in_channel=5)
    sagittal_t1 = RSNA24DatasetValid(df, train_des, image_dir, plane="Sagittal T1", use_cache=use_cache, in_channel=5)
    axial = RSNA24DatasetValid(df, train_des, image_dir, plane="Axial T2", use_cache=use_cache, in_channel=5)
    dataset = torch.utils.data.ConcatDataset([sagittal_t2, sagittal_t1, axial])
    for i in tqdm.tqdm(range(len(dataset)), leave=True):
        x, label = dataset.__getitem__(i)
        print(label)
        # print(x.shape)

    dataset = RSNA24DatasetTrain(df, train_des, image_dir, use_cache=use_cache, in_channel=5)
    for i in tqdm.tqdm(range(len(dataset)), leave=True):
        x, label = dataset.__getitem__(i)
        # print(x.shape)
    # plt.imshow(x[0], cmap="gray")
    # plt.show()


if __name__ == '__main__':
    main()
