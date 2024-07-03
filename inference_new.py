import math

import torch
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
import cv2

# Done: Submit 2 best models to see the actual position of the competition
#   DenseNet-201: ~20 million parameters - 0.66 - 0.70 - 0.63 - 0.70 - 0.62 - 0.75
#   MobileNetV3 Large: ~5.4 million parameters
#       mobilenetv3_large_100 - 0.74 - 0.77
# TODO Submit retrained models to check progress

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
        self.study_ids = list(df['study_id'].unique())
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
        # label_size = 0
        # # if x.series_description == "Sagittal T2/STIR":
        # #     # Spinal Canal Stenosis
        # #     # [l1/l2, l2/l3, l3/l4, l4/l5, l5/s1] * [Normal/Mild, Moderate, Severe]
        # #     label_size = 15
        # # if x.series_description == "Sagittal T1":
        # #     # Neural Foraminal Narrowing
        # #     # [left, right] * [l1/l2, l2/l3, l3/l4, l4/l5, l5/s1] * [Normal/Mild, Moderate, Severe]
        # #     label_size = 30
        # # if x.series_description == "Axial T2":
        # #     # Neural Foraminal Narrowing
        # #     # [left, right] * [Normal/Mild, Moderate, Severe]
        # #     label_size = 6
        # #
        # # label = np.zeros(label_size)
        # # for full_label in x[self.label]:
        # #     label[full_label] = 1
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

    y_preds = np.concatenate(y_preds, axis=0)
    data["preds"] = y_preds.tolist()

    return data


def prepare_submission(dataset, model_name, image_dir):
    sub = None
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
    sagittal_t1 = get_model_output(
        dataset.loc[dataset.series_description == "Sagittal T1"],
        OUTPUT_DIR + '/sagittal_t1-best_wll_model_fold-0.pt',
        30,
        image_dir
    )
    axial_t2 = get_model_output(
        dataset.loc[dataset.series_description == "Axial T2"],
        OUTPUT_DIR + '/axial_t2-best_wll_model_fold-0.pt',
        6,
        image_dir
    )
    print(sagittal_t2.head().to_string())
    print(sagittal_t1.head().to_string())
    print(axial_t2.head().to_string())

    # sub = pd.DataFrame()
    # sub['row_id'] = row_names
    # sub[LABELS] = y_preds
    return sub


if __name__ == '__main__':
    df = pd.read_csv(f'{rd}/test_series_descriptions.csv')
    submission = prepare_submission(df, MODEL_NAME, f"{rd}/test_images/")
    print(submission.shape)
    submission.to_csv('submission.csv', index=False)

    print(pd.read_csv('submission.csv').to_string())
