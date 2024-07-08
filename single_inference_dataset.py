import warnings
from pathlib import Path

import cv2
import numpy as np
import pydicom
from pandas.errors import SettingWithCopyWarning
from torch.utils.data import Dataset
import albumentations as A

# TODO OneHot output
# TODO Balanced dataset

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = Path("/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification")


def load_dicom(path):
    try:
        dicom = pydicom.read_file(path)
        data = dicom.pixel_array
        data = data - np.min(data)
        if np.max(data) != 0:
            data = data / np.max(data)
        data = (data * 255).astype(np.uint8)
    except Exception as e:
        print(e)
        data = np.zeros((200, 200)).astype(np.uint8)
    return data


def process_axial(transformed):
    transformed1 = A.Compose([
        A.InvertImg(always_apply=True),
    ])(image=transformed)["image"]

    transformed2 = A.Compose([
        A.Equalize(always_apply=True)
    ])(image=transformed)["image"]

    transformed3 = A.Compose([
        A.Crop(always_apply=False, p=1.0, x_min=100, y_min=100, x_max=220, y_max=220),
        A.Resize(300, 300),
    ])(image=transformed)["image"]

    transformed4 = A.Compose([
        A.InvertImg(always_apply=True),
    ])(image=transformed3)["image"]

    transformed5 = A.Compose([
        A.Equalize(always_apply=True)
    ])(image=transformed3)["image"]

    transformed6 = A.Compose([
        A.Crop(always_apply=False, p=1.0, x_min=150, y_min=100, x_max=240, y_max=220),
        A.Resize(300, 300)
    ])(image=transformed)["image"]

    transformed7 = A.Compose([
        A.InvertImg(always_apply=True),
    ])(image=transformed6)["image"]

    transformed8 = A.Compose([
        A.Equalize(always_apply=True),
    ])(image=transformed6)["image"]

    return np.vstack((
        np.hstack((transformed, transformed1, transformed2)),
        np.hstack((transformed3, transformed4, transformed5)),
        np.hstack((transformed6, transformed7, transformed8))
    ))


def process_sagittal(transformed):
    transformed1 = A.Compose([
        A.InvertImg(always_apply=True),
    ])(image=transformed)["image"]

    transformed2 = A.Compose([
        A.Sharpen(always_apply=True, alpha=(0.2, 0.2), lightness=(3, 3), p=0.75)
    ])(image=transformed)["image"]

    transformed3 = A.Compose([
        A.Crop(always_apply=True, p=1.0, x_min=40, y_min=40, x_max=200, y_max=240),
        A.Resize(300, 300),
    ])(image=transformed)["image"]

    transformed4 = A.Compose([
        A.InvertImg(always_apply=True),
    ])(image=transformed3)["image"]

    transformed5 = A.Compose([
        A.Sharpen(always_apply=True, alpha=(0.2, 0.2), lightness=(3, 3), p=0.75)
    ])(image=transformed3)["image"]

    transformed6 = A.Compose([
        A.Equalize(always_apply=True)
    ])(image=transformed3)["image"]

    transformed7 = A.Compose([
        A.Equalize(always_apply=True)
    ])(image=transformed)["image"]

    transformed8 = A.Compose([
        A.Downscale(always_apply=False, scale_min=0.099, scale_max=0.099)
    ])(image=transformed3)["image"]

    return np.vstack((
        np.hstack((transformed, transformed1, transformed2)),
        np.hstack((transformed3, transformed4, transformed5)),
        np.hstack((transformed6, transformed7, transformed8))
    ))


def validation_transform(height, width):
    return A.Compose([
            A.Resize(height, width),
            A.Normalize(mean=0.5, std=0.5),
            A.ToRGB()
        ])


def resize_transform():
    return A.Compose([
        A.Resize(300, 300),
    ])


class RSNA24DatasetInference(Dataset):
    def __init__(self, dataframe, image_dir, image_size):
        self.df = dataframe
        self.study_ids = list(self.df['study_id'].unique())
        self.image_dir = image_dir
        self.resize = resize_transform()
        self.common_transform = validation_transform(image_size[0], image_size[1])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.iloc[idx]
        image = load_dicom(f"{self.image_dir}/{x['study_id']}/{x['series_id']}/{x['instance_number']}.dcm")

        image = self.resize(image=image)["image"]

        if x.series_description == "Sagittal T2/STIR" or x.series_description == "Sagittal T1":
            image = process_sagittal(image)
        if x.series_description == "Axial T2":
            image = process_axial(image)

        image = self.common_transform(image=image)["image"]
        image = image.transpose(2, 0, 1).astype(np.float32)

        return image


class RSNA24DatasetActivation(Dataset):
    def __init__(self, st_ids, sagittal_t2_feat, sagittal_t1_feat, axial_t2_feat):
        self.sagittal_t2_feat = sagittal_t2_feat
        self.sagittal_t1_feat = sagittal_t1_feat
        self.axial_t2_feat = axial_t2_feat

        self.study_ids = st_ids

        self.image_size = (240, 240)
        self.label = "full_label"

    def __len__(self):
        return len(self.study_ids)

    def create_image(self, data):
        data = data.sort_values("instance_number")
        data = [np.array(i) for i in data.preds.values]
        return cv2.resize(np.array(data), self.image_size, interpolation=cv2.INTER_AREA).transpose((2, 0, 1))

    def __getitem__(self, i):
        study_id = self.study_ids[i]
        xs2 = self.create_image(self.sagittal_t2_feat.loc[self.sagittal_t2_feat.study_id == study_id])
        xs1 = self.create_image(self.sagittal_t1_feat.loc[self.sagittal_t1_feat.study_id == study_id])
        xt1 = self.create_image(self.axial_t2_feat.loc[self.axial_t2_feat.study_id == study_id])
        feat = np.vstack((xs2, xs1, xt1))
        return feat.astype(np.float32), study_id
