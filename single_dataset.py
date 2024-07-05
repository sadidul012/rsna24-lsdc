import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import tqdm
from pandas.errors import SettingWithCopyWarning
from torch.utils.data import Dataset
import albumentations as A
import pickle
from single_inference import load_dicom, process_sagittal, process_axial, resize_transform, validation_transform

# TODO OneHot output
# TODO Balanced dataset

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

DATA_PATH = Path("/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification")


def get_full_label():
    plane_conditions = {
        "Sagittal T2/STIR": ["Spinal Canal Stenosis"],
        "Sagittal T1": ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing"],
        "Axial T2": ["Left Subarticular Stenosis", "Right Subarticular Stenosis"]
    }
    levels = ["l1/l2", "l2/l3", "l3/l4", "l4/l5", "l5/s1"]
    labels = ["Normal/Mild", "Moderate", "Severe"]

    full_labels = {}

    index_count = 0
    _full_labels = {}
    for c in plane_conditions["Sagittal T2/STIR"]:
        for level in levels:
            for label in labels:
                _full_labels[(c + "_" + level + "_" + label).replace(" ", "_").replace("/", "_").lower()] = index_count
                index_count += 1
    full_labels["Sagittal T2/STIR"] = _full_labels

    index_count = 0
    _full_labels = {}
    for c in plane_conditions["Sagittal T1"]:
        for level in levels:
            for label in labels:
                _full_labels[(c + "_" + level + "_" + label).replace(" ", "_").replace("/", "_").lower()] = index_count
                index_count += 1
    full_labels["Sagittal T1"] = _full_labels

    index_count = 0
    _full_labels = {}
    for c in plane_conditions["Axial T2"]:
        for label in labels:
            _full_labels[(c + "_" + label).replace(" ", "_").replace("/", "_").lower()] = index_count
            index_count += 1
    full_labels["Axial T2"] = _full_labels
    return full_labels


def read_train_csv(data_path):
    if os.path.exists(data_path / "temp_train.csv") and os.path.exists(data_path / "temp_train_solution.csv"):
        return pd.read_csv(data_path / "temp_train.csv"), pd.read_csv(data_path / "temp_train_solution.csv")

    train_main = pd.read_csv(data_path / "train.csv")

    solution = train_main.melt(id_vars=["study_id"], var_name="full_label", value_name="severity")
    solution["row_id"] = solution.apply(lambda row: str(row.study_id) + "_" + row.full_label, axis=1)
    solution.severity = solution.severity.fillna("Normal/Mild")
    solution.loc[solution.severity == "Normal/Mild", "normal_mild"] = 1
    solution.loc[solution.severity == "Moderate", "moderate"] = 1
    solution.loc[solution.severity == "Severe", "severe"] = 1

    solution.loc[solution.severity == "Normal/Mild", "sample_weight"] = 1
    solution.loc[solution.severity == "Moderate", "sample_weight"] = 2
    solution.loc[solution.severity == "Severe", "sample_weight"] = 3

    solution = solution[["study_id", "row_id", "normal_mild", "moderate", "severe", "sample_weight"]]
    solution = solution.fillna(0)
    solution = solution.sort_values(by=["row_id"])
    solution.to_csv(data_path / "temp_train_solution.csv", index=False)

    train_desc = pd.read_csv(data_path / "train_series_descriptions.csv")
    train_label_coordinates = pd.read_csv(data_path / "train_label_coordinates.csv")
    for i, row in train_desc.iterrows():
        train_label_coordinates.loc[
            (train_label_coordinates.study_id == row.study_id) & (train_label_coordinates.series_id == row.series_id),
            "plane"
        ] = row.series_description

    train_label_coordinates["label"] = train_label_coordinates.apply(
        lambda row: train_main.loc[train_main.study_id == row.study_id][
            (row.condition + '_' + row.level).replace(" ", "_").replace("/", "_").lower()
        ].reset_index(drop=True).iloc[0],
        axis=1
    )
    train_label_coordinates.to_csv(data_path / "temp_train.csv", index=False)

    return train_label_coordinates, solution


def train_transform(aug_prob=0.75):
    return A.Compose([
        A.Resize(240, 240),
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=aug_prob),
        A.OneOf([
            A.MedianBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=aug_prob),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0.01, border_mode=0, p=aug_prob),
        A.Superpixels(p_replace=(0, 0.1), n_segments=(100, 100), max_size=128, p=aug_prob),
        A.GridDistortion(num_steps=5, distort_limit=(-0.5, 0.5), interpolation=0, border_mode=4, p=0.75),
        A.Resize(300, 300),
    ])


def process_train_csv(train):
    train.loc[(train.plane == "Sagittal T1") & (train.condition == "Spinal Canal Stenosis"), "plane"] = "Sagittal T2/STIR"
    full_labels = get_full_label()
    sagittal_t2 = train.loc[train.plane == "Sagittal T2/STIR"]
    sagittal_t2.loc[:, "label"] = sagittal_t2["label"].fillna("Normal/Mild")
    sagittal_t2.loc[:, "full_label"] = sagittal_t2.apply(
        lambda x: (x.condition + "_" + x.level + "_" + x.label).replace("/", "_").replace(" ", "_").lower(), axis=1)
    sagittal_t2.loc[:, "full_label"] = sagittal_t2["full_label"].apply(lambda x: full_labels["Sagittal T2/STIR"][x])
    sagittal_t2 = sagittal_t2.groupby(["study_id", "series_id", "instance_number"]).agg({
        # "condition": pd.Series.to_list,
        # "level": "first",
        # "x": pd.Series.to_list,
        # "y": pd.Series.to_list,
        "plane": "first",
        # "label": pd.Series.to_list,
        "full_label": pd.Series.to_list
    }).reset_index(drop=False)

    sagittal_t1 = train.loc[train.plane == "Sagittal T1"]
    sagittal_t1.loc[:, "label"] = sagittal_t1["label"].fillna("Normal/Mild")
    sagittal_t1.loc[:, "full_label"] = sagittal_t1.apply(
        lambda x: (x.condition + "_" + x.level + "_" + x.label).replace("/", "_").replace(" ", "_").lower(), axis=1)
    sagittal_t1.loc[:, "full_label"] = sagittal_t1["full_label"].apply(lambda x: full_labels["Sagittal T1"][x])
    sagittal_t1 = sagittal_t1.groupby(["study_id", "series_id", "instance_number"]).agg({
        # "condition": pd.Series.to_list,
        # "level": "first",
        # "x": pd.Series.to_list,
        # "y": pd.Series.to_list,
        "plane": "first",
        # "label": pd.Series.to_list,
        "full_label": pd.Series.to_list
    }).reset_index(drop=False)

    axial_t2 = train.loc[train.plane == "Axial T2"]
    axial_t2.loc[:, "label"] = axial_t2["label"].fillna("Normal/Mild")
    axial_t2.loc[:, "full_label"] = axial_t2.apply(
        lambda x: (x.condition + "_" + x.label).replace("/", "_").replace(" ", "_").lower(), axis=1)
    axial_t2.loc[:, "full_label"] = axial_t2["full_label"].apply(lambda x: full_labels["Axial T2"][x])
    axial_t2 = axial_t2.groupby(["study_id", "series_id", "instance_number"]).agg({
        # "condition": pd.Series.to_list,
        # "level": "first",
        # "x": pd.Series.to_list,
        # "y": pd.Series.to_list,
        "plane": "first",
        # "label": pd.Series.to_list,
        "full_label": pd.Series.to_list
    }).reset_index(drop=False)

    return sagittal_t2, sagittal_t1, axial_t2


def balance(data):
    g = data.groupby(["plane", "condition", "level", "label"])
    g = g.apply(lambda x: x.sample(250, replace=len(x) <= 250).reset_index(drop=True))
    data = g.reset_index(drop=True)
    return data


class RSNA24DatasetBase(Dataset):
    def __init__(self, dataframe, transform=None, label_column='full_label', use_cache=True, split="train", image_size=(512, 512)):
        self.dataframe = dataframe
        self.transform = transform
        self.label = label_column
        self.use_cache = use_cache
        self.split = split
        self.common_transform = validation_transform(image_size[0], image_size[1])
        self.resize = resize_transform()

    def __len__(self):
        return len(self.dataframe)

    def save_instance(self, value, idx):
        path = f"/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification/temp_2/{idx}.bz2"
        if self.use_cache:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(value, f)

    def load_instance(self, idx):
        path = f"/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification/temp_2/{idx}.bz2"
        if self.use_cache:
            if os.path.exists(path):
                with open(path, "rb") as file:
                    return pickle.load(file)

        return None

    def read_image(self, x):
        image = self.load_instance(f"{x["study_id"]}_{x["series_id"]}_{x["instance_number"]}")
        if image is not None:
            return image

        image = load_dicom(DATA_PATH / f"train_images/{x["study_id"]}/{x["series_id"]}/{x["instance_number"]}.dcm")
        self.save_instance(image, f"{x["study_id"]}_{x["series_id"]}_{x["instance_number"]}")

        return image

    def __getitem__(self, item):
        x = self.dataframe.iloc[item]

        label_size = 0
        if x.plane == "Sagittal T2/STIR":
            # Spinal Canal Stenosis
            # [l1/l2, l2/l3, l3/l4, l4/l5, l5/s1] * [Normal/Mild, Moderate, Severe]
            label_size = 15
        if x.plane == "Sagittal T1":
            # Neural Foraminal Narrowing
            # [left, right] * [l1/l2, l2/l3, l3/l4, l4/l5, l5/s1] * [Normal/Mild, Moderate, Severe]
            label_size = 30
        if x.plane == "Axial T2":
            # Neural Foraminal Narrowing
            # [left, right] * [Normal/Mild, Moderate, Severe]
            label_size = 6

        label = np.zeros(label_size)
        for full_label in x[self.label]:
            label[full_label] = 1

        # label = label.reshape(-1, 3)
        image = self.read_image(x)

        if self.transform:
            image = self.transform(image=image)["image"]
        else:
            image = self.resize(image=image)["image"]

        if x.plane == "Sagittal T2/STIR" or x.plane == "Sagittal T1":
            image = process_sagittal(image)
        if x.plane == "Axial T2":
            image = process_axial(image)

        image = self.common_transform(image=image)["image"]
        image = image.transpose(2, 0, 1).astype(np.float32)

        return image, label.astype(np.float32)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    _train, sol = read_train_csv(DATA_PATH)
    print(sol.shape)
    # balanced = balance(train)
    print(_train.shape)

    _sagittal_t2, _sagittal_t1, _axial_t2 = process_train_csv(_train)
    print("_sagittal_t2", _sagittal_t2.shape)

    # dataset = RSNA24DatasetBase(_sagittal_t2, transform=validation_transform(512, 512))
    # for i in tqdm.tqdm(range(len(dataset))):
    #     _x, _label = dataset.__getitem__(i)
    #     print(_x.shape, _label.shape)
    #     break
    #
    # print("_sagittal_t1", _sagittal_t1.shape)
    #
    dataset = RSNA24DatasetBase(_sagittal_t1)
    for i in tqdm.tqdm(range(len(dataset))):
        _x, _label = dataset.__getitem__(i)
        print(_x.shape, _label.shape)
        break

    print("_axial_t2", _axial_t2.shape)
    dataset = RSNA24DatasetBase(_axial_t2, transform=train_transform(0.75), image_size=(1024, 1024))
    for i in tqdm.tqdm(range(len(dataset))):
        _x, _label = dataset.__getitem__(i)
        print(_x.shape, _label.shape)
        # plt.imshow(_x, cmap="gray")
        # plt.show()
        break
