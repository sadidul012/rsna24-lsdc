import math
import warnings

import torch
from pandas.errors import SettingWithCopyWarning
from torch import nn
from torch.utils.data import DataLoader
import timm
import glob
import os
import numpy as np
import pandas as pd
from glob import glob
import tqdm
from torch.utils.data import Dataset
import albumentations as A
import pydicom

# TODO Submit trained models to check progress

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

rd = '/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification'
OUTPUT_DIR = 'rsna24-data/xception41-c3p1b16e20f14'
MODEL_NAME = "xception41"

N_WORKERS = math.floor(os.cpu_count()/2) + 1
USE_AMP = True
SEED = 8620

IMG_SIZE = [512, 512]
IN_CHANS = 3

BATCH_SIZE = 32
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)

sample_sub = pd.read_csv(f'{rd}/sample_submission.csv')
LABELS = list(sample_sub.columns[1:])
CONDITIONS = [
    'spinal_canal_stenosis',
    'left_neural_foraminal_narrowing',
    'right_neural_foraminal_narrowing',
    'left_subarticular_stenosis',
    'right_subarticular_stenosis'
]

LEVELS = [
    'l1_l2',
    'l2_l3',
    'l3_l4',
    'l4_l5',
    'l5_s1',
]

plane_conditions = {
    "Sagittal T2/STIR": ["spinal_canal_stenosis"],
    "Sagittal T1": ["left_neural_foraminal_narrowing", "right_neural_foraminal_narrowing"],
    "Axial T2": ["left_subarticular_stenosis", "right_subarticular_stenosis"]
}
levels = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]
row_names = {
    "Sagittal T2/STIR": [],
    "Sagittal T1": [],
    "Axial T2": []
}
for k in plane_conditions:
    for p in plane_conditions[k]:
        for label in levels:
            row_names[k].append(f"{p}_{label}")

labels = ["row_id", "normal_mild", "moderate", "severe"]


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
    def __init__(self, dataframe, image_dir):
        self.df = dataframe
        self.study_ids = list(self.df['study_id'].unique())
        self.image_dir = image_dir
        self.resize = resize_transform()
        self.common_transform = validation_transform(IMG_SIZE[0], IMG_SIZE[1])

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


class RSNA24Model(nn.Module):
    def __init__(self, model_name, in_c=3, n_classes=30, pretrained=True, features_only=False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=features_only,
            in_chans=in_c,
            num_classes=n_classes,
            global_pool='avg'
        )
        # print(torchinfo.summary(self.model, (IN_CHANS, *IMG_SIZE)))
        print(f"Params: {(torch.nn.utils.parameters_to_vector(self.model.parameters()).numel()/1000000):.2f}M")

    def forward(self, x):
        y = self.model(x)
        return y


def get_model_output(data, path, n_classes, image_dir, model_name):
    model = RSNA24Model(model_name, IN_CHANS, n_classes, pretrained=False)
    model.load_state_dict(torch.load(path))
    model.eval()
    model.half()
    model.to(device)

    dataset = RSNA24DatasetInference(data, image_dir)
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS, pin_memory=False, drop_last=False)

    y_preds = []
    with tqdm.tqdm(total=len(dl), leave=True) as pbar:
        with torch.no_grad():
            for idx, x in enumerate(dl):
                x = x.to(device).type(torch.float32)
                with autocast:
                    y = model(x).reshape(-1, int(n_classes/3), 3).softmax(2)
                    y = y.cpu().numpy()
                    y_preds.append(y)

                pbar.update()

    y_preds = np.concatenate(y_preds, axis=0)
    data["preds"] = y_preds.tolist()
    return data


def apply(x):
    preds = []
    if x.iloc[0].series_description == "Sagittal T2/STIR":
        preds = np.array([np.array(y) for y in x["preds"].values]).mean(0)

    if x.iloc[0].series_description == "Sagittal T1":
        preds = np.array([np.array(y) for y in x["preds"].values]).mean(0)

    if x.iloc[0].series_description == "Axial T2":
        x['instance_number'] = x['instance_number'].astype('int32')
        x = x.sort_values(by=["instance_number"])
        step = math.floor(x.shape[0]/5)
        left = []
        right = []
        for i in range(0, 5):
            y = x.iloc[i*step:(i * step)+step]
            preds = np.array([np.array(z) for z in y["preds"].values]).mean(0)
            left.append(preds[0])
            right.append(preds[1])

        preds = np.array(left + right)

    result = []
    for level, pred in zip(row_names[x.iloc[0].series_description], preds):
        result.append([f"{x.iloc[0]['study_id']}_{level}"] + list(pred))
    result = pd.DataFrame(result, columns=labels)

    return result


def inject_series_description(x):
    rows = pd.DataFrame([
        [x.iloc[0].study_id, -100, "Axial T2"],
        [x.iloc[0].study_id, -100, 'Sagittal T2/STIR'],
        [x.iloc[0].study_id, -100, "Sagittal T1"]
    ], columns=["study_id", "series_id", "series_description"])

    rows = pd.concat([x, rows]).drop_duplicates(subset=["study_id", "series_description"], keep="first")
    return rows


def instance_image_path(x, image_dir):
    return [
        os.path.splitext(os.path.basename(d))[0] for d in glob(f"{image_dir}/{x['study_id']}/{x['series_id']}/*.dcm")
    ]


def prepare_submission(dataset, image_dir, sagittal_model_t2, sagittal_model_t1, axial_model_t1, model_name):
    # TODO utilize all data
    dataset = dataset.drop_duplicates(subset=["study_id", "series_description"])
    dataset = dataset.groupby("study_id").apply(lambda x: inject_series_description(x)).reset_index(drop=True)
    dataset["instance_number"] = dataset.apply(lambda x: instance_image_path(x, image_dir), axis=1)
    dataset = dataset.explode("instance_number")

    sagittal_t2 = get_model_output(
        dataset.loc[dataset.series_description == "Sagittal T2/STIR"],
        sagittal_model_t2,
        15,
        image_dir,
        model_name
    )
    sagittal_t2 = sagittal_t2.groupby(by=["study_id", "series_id"]).apply(lambda x: apply(x)).reset_index(drop=True)

    sagittal_t1 = get_model_output(
        dataset.loc[dataset.series_description == "Sagittal T1"],
        sagittal_model_t1,
        30,
        image_dir,
        model_name
    )
    sagittal_t1 = sagittal_t1.groupby(by=["study_id", "series_id"]).apply(lambda x: apply(x)).reset_index(drop=True)

    axial_t2 = get_model_output(
        dataset.loc[dataset.series_description == "Axial T2"],
        axial_model_t1,
        6,
        image_dir,
        model_name
    )
    axial_t2 = axial_t2.groupby(by=["study_id", "series_id"]).apply(lambda x: apply(x)).reset_index(drop=True)
    sub = pd.concat([sagittal_t1, axial_t2, sagittal_t2]).sort_values("row_id")
    # sub = sub.groupby("study_id").apply(lambda x: x).reset_index(drop=True)[["row_id", "normal_mild", "moderate", "severe"]]
    # print(sub.to_string())
    return sub.reset_index(drop=True)


# 2.3338262493296744
if __name__ == '__main__':
    df = pd.read_csv(f'{rd}/test_series_descriptions.csv')
    submission = prepare_submission(
        df,
        f"{rd}/test_images/",
        OUTPUT_DIR + '/sagittal_t2-best_wll_model_fold-0.pt',
        OUTPUT_DIR + '/sagittal_t1-best_wll_model_fold-0.pt',
        OUTPUT_DIR + '/axial_t2-best_wll_model_fold-0.pt',
        MODEL_NAME
    )
    print(submission.shape)
    submission.to_csv('submission.csv', index=False)

    print(pd.read_csv('submission.csv').head().to_string())
