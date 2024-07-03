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

rd = '/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification'
OUTPUT_DIR = f'rsna24-data/rsna24-3-efficientnet_b2-5'
MODEL_NAME = "efficientnet_b2"

N_WORKERS = math.floor(os.cpu_count()/3) + 1
USE_AMP = True
SEED = 8620

IMG_SIZE = [512, 512]
IN_CHANS = 3

BATCH_SIZE = 1
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

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


def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


class RSNA24DatasetInference(Dataset):
    def __init__(self, dataframe, image_dir):
        self.df = dataframe
        self.study_ids = list(self.df['study_id'].unique())
        self.image_dir = image_dir
        self.transform = A.Compose([
            A.Resize(IMG_SIZE[0], IMG_SIZE[1]),
            A.Normalize(mean=0.5, std=0.5),
            A.ToRGB()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = self.df.iloc[idx]
        image = load_dicom(f"{self.image_dir}/{x["study_id"]}/{x["series_id"]}/{x["instance_number"]}.dcm")
        if self.transform:
            image = self.transform(image=image)["image"]
            image = image.transpose(2, 0, 1).astype(np.float32)

        return image


class RSNA24Model(nn.Module):
    def __init__(self, model_name, in_c=30, n_classes=75, pretrained=True, features_only=False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=features_only,
            in_chans=in_c,
            num_classes=n_classes,
            global_pool='avg'
        )

    def forward(self, x):
        y = self.model(x)
        return y


autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)


def get_model_output(data, path, n_classes, image_dir):
    model = RSNA24Model(MODEL_NAME, IN_CHANS, n_classes, pretrained=False)
    model.load_state_dict(torch.load(path))
    model.eval()
    model.half()
    model.to(device)

    dataset = RSNA24DatasetInference(data, image_dir)
    dl = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=N_WORKERS, pin_memory=False, drop_last=False)

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
        result.append([f"{x.iloc[0]["study_id"]}_{level}"] + list(pred))
    result = pd.DataFrame(result, columns=labels)

    return result


def prepare_submission(dataset, image_dir):
    # TODO utilize all data
    dataset = dataset.drop_duplicates(subset=["study_id", "series_description"])
    dataset["instance_number"] = dataset.apply(lambda x: [
        os.path.splitext(os.path.basename(d))[0] for d in glob(f"{image_dir}/{x["study_id"]}/{x["series_id"]}/*.dcm")
    ], axis=1)

    dataset = dataset.explode("instance_number")

    sagittal_t2 = get_model_output(
        dataset.loc[dataset.series_description == "Sagittal T2/STIR"],
        OUTPUT_DIR + '/sagittal_t2-best_wll_model_fold-0.pt',
        15,
        image_dir
    )
    sagittal_t2 = sagittal_t2.groupby(by=["study_id", "series_id"]).apply(lambda x: apply(x)).reset_index(drop=True)

    sagittal_t1 = get_model_output(
        dataset.loc[dataset.series_description == "Sagittal T1"],
        OUTPUT_DIR + '/sagittal_t1-best_wll_model_fold-0.pt',
        30,
        image_dir
    )
    sagittal_t1 = sagittal_t1.groupby(by=["study_id", "series_id"]).apply(lambda x: apply(x)).reset_index(drop=True)

    axial_t2 = get_model_output(
        dataset.loc[dataset.series_description == "Axial T2"],
        OUTPUT_DIR + '/axial_t2-best_wll_model_fold-0.pt',
        6,
        image_dir
    )
    axial_t2 = axial_t2.groupby(by=["study_id", "series_id"]).apply(lambda x: apply(x)).reset_index(drop=True)
    sub = pd.concat([sagittal_t2, sagittal_t1, axial_t2])
    # sub = sub.sort_values(by="row_id")
    return sub.reset_index(drop=True)


if __name__ == '__main__':
    df = pd.read_csv(f'{rd}/test_series_descriptions.csv')
    submission = prepare_submission(df, f"{rd}/test_images/")
    print(submission.shape)
    submission.to_csv('submission.csv', index=False)

    print(pd.read_csv('submission.csv').to_string())
