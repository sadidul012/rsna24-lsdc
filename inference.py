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
OUTPUT_DIR = f'rsna24-data/rsna24-new-densenet201-5'
MODEL_NAME = "densenet201"

N_WORKERS = os.cpu_count()
USE_AMP = True
SEED = 8620

IMG_SIZE = [512, 512]
IN_CHANS = 30
N_LABELS = 25
N_CLASSES = 3 * N_LABELS


BATCH_SIZE = 1
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

df = pd.read_csv(f'{rd}/test_series_descriptions.csv')
study_ids = list(df['study_id'].unique())
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
    # if reverse_sort=False, then increasing array index will be from RIGHT->LEFT and CAUDAL->CRANIAL
    # thus we do reverse_sort=True for axial so increasing array index is craniocaudal
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


class RSNA24DatasetInference(Dataset):
    def __init__(self, df, image_dir, in_channel=30, image_size=(512, 512), split="valid"):
        self.df = df
        self.study_ids = list(df['study_id'].unique())
        self.image_dir = image_dir
        self.image_size = image_size
        self.in_channel = in_channel
        self.split = split
        self.planes = {"Sagittal T2/STIR": "sagittal", "Sagittal T1": "sagittal", "Axial T2": "axial"}

        self.df = self.df.fillna(-100)
        self.label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
        self.df = self.df.replace(self.label2id)

        self.transform = A.Compose([
            A.Resize(self.image_size[0], self.image_size[1]),
            A.Normalize(mean=0.5, std=0.5)
        ])

    def __len__(self):
        return len(self.study_ids)

    def process(self, data, return_length, offset=0, axial=False):
        meta_height = int(self.image_size[0] / (self.in_channel - 1) * return_length)

        if data is None:
            return np.zeros((return_length, self.image_size[0], self.image_size[1])), np.zeros(
                (meta_height, self.image_size[1]))

        if axial:
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
            positions = np.array(positions)[:return_length]

        else:
            middle = len(data["array"]) // 2
            start = min(middle - offset - (return_length // 2), len(data["array"]) - return_length)
            start = start if start >= 0 else 0
            end = min(middle - offset + (return_length // 2), len(data["array"]))
            if return_length % 2 != 0:
                end += 1

            images = data["array"][start:end]
            positions = data["positions"][start:end]

        pixel_spacing = data["pixel_spacing"].reshape(-1, 2)
        pixel_spacing = np.pad(pixel_spacing, ((0, 0), (0, 1)), mode="constant", constant_values=0)
        positions = np.concatenate((positions, pixel_spacing))
        meta_data = (
            np.pad(
                positions,
                ((abs(positions.shape[0] - meta_height), 0), (0, abs(positions.shape[1] - self.image_size[1]))),
                mode='constant', constant_values=0
            )
        )

        images = images.transpose(2, 1, 0)
        if self.transform is not None:
            images = self.transform(image=images)['image']
        images = images.transpose(2, 1, 0)

        if images.shape[0] - return_length < 0:
            images = np.pad(
                images, ((abs(images.shape[0] - return_length), 0), (0, 0), (0, 0)), mode='constant', constant_values=0
            )
        return images, meta_data

    def __getitem__(self, idx):
        st_id = self.study_ids[idx]
        study = self.df.loc[self.df.study_id == st_id]
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
        sag_t1_images, sag_t1_meta = self.process(data_["Sagittal T1"], 10, 1)
        sag_t2_images, sag_t2_meta = self.process(data_["Sagittal T2/STIR"], 9, -4)
        ax_t2_images, ax_t2_meta = self.process(data_["Axial T2"], 10, axial=True)
        meta = np.concatenate((sag_t1_meta, sag_t2_meta, ax_t2_meta), axis=0)
        meta = np.pad(
            meta,
            ((abs(meta.shape[0] - self.image_size[0]), 0), (0, 0)), mode='constant', constant_values=0
        )
        x = np.concatenate((sag_t1_images, sag_t2_images, ax_t2_images, [meta]), axis=0)
        return x, str(st_id)


test_ds = RSNA24DatasetInference(df, f"{rd}/test_images/")
test_dl = DataLoader(
    test_ds,
    batch_size=1,
    shuffle=False,
    num_workers=N_WORKERS,
    pin_memory=True,
    drop_last=False
)


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


models = []
CKPT_PATHS = glob(OUTPUT_DIR + '/best_wll_model_fold-*.pt')
CKPT_PATHS = sorted(CKPT_PATHS)
for _, cp in enumerate(CKPT_PATHS):
    print(f'loading {cp}...')
    model = RSNA24Model(MODEL_NAME, IN_CHANS, N_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(cp))
    model.eval()
    model.half()
    model.to(device)
    models.append(model)

autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)
y_preds = []
row_names = []

with tqdm.tqdm(test_dl, leave=True) as pbar:
    with torch.no_grad():
        for idx, (x, si) in enumerate(pbar):
            x = x.to(device).type(torch.float32)
            pred_per_study = np.zeros((25, 3))

            for cond in CONDITIONS:
                for level in LEVELS:
                    row_names.append(si[0] + '_' + cond + '_' + level)

            with autocast:
                for m in models:
                    y = m(x)[0]
                    for col in range(N_LABELS):
                        pred = y[col * 3:col * 3 + 3]
                        y_pred = pred.float().softmax(0).cpu().numpy()
                        pred_per_study[col] += y_pred / len(models)

                y_preds.append(pred_per_study)

                # one hot output
                # y_hat = np.zeros(pred_per_study.shape)
                # ys = pred_per_study.argmax(axis=1)
                # for i, y in enumerate(ys):
                #     y_hat[i][y] = 1
                # y_preds.append(y_hat)

y_preds = np.concatenate(y_preds, axis=0)
sub = pd.DataFrame()
sub['row_id'] = row_names
sub[LABELS] = y_preds
print(sub.shape)
sub.to_csv('submission.csv', index=False)

print(pd.read_csv('submission.csv').head().to_string())
