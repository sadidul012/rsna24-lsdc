import math
import warnings

import torch
from pandas.errors import SettingWithCopyWarning
from torch.utils.data import DataLoader
import glob
import os
import numpy as np
import pandas as pd
from glob import glob
import tqdm
from single_dataset import RSNA24DatasetInference, RSNA24DatasetActivation
from single_model import RSNA24Model

from config import ModelConfig

# TODO Submit trained models to check progress

warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

rd = '/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification'
sagittal_t2_model_config = ModelConfig("rsna24-data/models_db/xception41-DB-c3p1b16e2f14/axial_t2-best_wll_model_fold-0.json")
sagittal_t1_model_config = ModelConfig("rsna24-data/models_db/xception41-DB-c3p1b16e2f14/sagittal_t1-best_wll_model_fold-0.json")
axial_t2_model_config = ModelConfig("rsna24-data/models_db/xception41-DB-c3p1b16e2f14/sagittal_t2-best_wll_model_fold-0.json")
activation_model_config = ModelConfig("rsna24-data/models/rexnet_150.nav_in1k-A-c9p1b16e20f14/Activation-best_wll_model_fold-0.json")

N_WORKERS = math.floor(os.cpu_count()/2) + 1
USE_AMP = True
SEED = 8620

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


def get_model_output(data, config: ModelConfig, image_dir):
    model = RSNA24Model(config.MODEL_NAME, config.IN_CHANS, config.N_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(config.MODEL_PATH + "/" + config.MODEL_FILENAME))
    model.eval()
    model.half()
    model.to(device)

    dataset = RSNA24DatasetInference(data, image_dir, config.IMG_SIZE)
    dl = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=N_WORKERS, pin_memory=False, drop_last=False)

    y_preds = []
    with tqdm.tqdm(total=len(dl), leave=True) as pbar:
        with torch.no_grad():
            for idx, x in enumerate(dl):
                x = x.to(device).type(torch.float32)
                with autocast:
                    y = model(x).reshape(-1, config.N_LABELS, 3).softmax(2)
                    y = y.cpu().numpy()
                    y_preds.append(y)

                pbar.update()

    y_preds = np.concatenate(y_preds, axis=0)
    data["preds"] = y_preds.tolist()
    return data


def apply_average(x):
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


def average_method(sagittal_t2, sagittal_t1, axial_t2):
    sagittal_t2 = sagittal_t2.groupby(by=["study_id", "series_id"]).apply(lambda x: apply_average(x)).reset_index(drop=True)
    sagittal_t1 = sagittal_t1.groupby(by=["study_id", "series_id"]).apply(lambda x: apply_average(x)).reset_index(drop=True)
    axial_t2 = axial_t2.groupby(by=["study_id", "series_id"]).apply(lambda x: apply_average(x)).reset_index(drop=True)

    return pd.concat([sagittal_t1, axial_t2, sagittal_t2]).sort_values("row_id")


def activation_method(dataset, config: ModelConfig, sagittal_t2, sagittal_t1, axial_t2):
    test_ds = RSNA24DatasetActivation(
        dataset.study_id.unique(),
        sagittal_t2,
        sagittal_t1,
        axial_t2,
    )
    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, pin_memory=False, drop_last=False, num_workers=N_WORKERS
    )
    model = RSNA24Model(config.MODEL_NAME, in_c=config.IN_CHANS, n_classes=config.N_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(config.MODEL_PATH + "/" + config.MODEL_FILENAME))
    model.eval()
    model.half()
    model.to(device)

    y_preds = []
    row_names = []
    with tqdm.tqdm(test_dl, leave=True) as pbar:
        with torch.no_grad():
            for idx, (x, study_id) in enumerate(pbar):
                x = x.to(device)
                for cond in CONDITIONS:
                    for level in LEVELS:
                        row_names.append(str(study_id[0].item()) + '_' + cond + '_' + level)

                pred_per_study = np.zeros((25, 3))
                with autocast:
                    y = model(x)[0]
                    for col in range(config.N_LABELS):
                        pred = y[col * 3:col * 3 + 3]
                        y_pred = pred.float().softmax(0).cpu().numpy()
                        pred_per_study[col] += y_pred
                y_preds.append(pred_per_study)
    y_preds = np.concatenate(y_preds, axis=0)
    sub = pd.DataFrame()
    sub['row_id'] = row_names
    sub[LABELS] = y_preds
    return sub


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


def prepare_submission(dataset, image_dir, activation_model, sagittal_model_t2, sagittal_model_t1, axial_model_t1, method="average"):
    # TODO utilize all data
    dataset = dataset.drop_duplicates(subset=["study_id", "series_description"])
    dataset = dataset.groupby("study_id").apply(lambda x: inject_series_description(x)).reset_index(drop=True)
    dataset["instance_number"] = dataset.apply(lambda x: instance_image_path(x, image_dir), axis=1)
    dataset = dataset.explode("instance_number")

    sagittal_t2 = get_model_output(
        dataset.loc[dataset.series_description == "Sagittal T2/STIR"],
        sagittal_model_t2,
        image_dir
    )

    sagittal_t1 = get_model_output(
        dataset.loc[dataset.series_description == "Sagittal T1"],
        sagittal_model_t1,
        image_dir
    )

    axial_t2 = get_model_output(
        dataset.loc[dataset.series_description == "Axial T2"],
        axial_model_t1,
        image_dir
    )
    if method == "activation":
        sub = activation_method(dataset, activation_model, sagittal_t2, sagittal_t1, axial_t2)
    else:
        sub = average_method(sagittal_t2, sagittal_t1, axial_t2)

    return sub.reset_index(drop=True)


if __name__ == '__main__':
    df = pd.read_csv(f'{rd}/test_series_descriptions.csv')
    submission = prepare_submission(
        df,
        f"{rd}/test_images/",
        activation_model_config,
        sagittal_t2_model_config,
        sagittal_t1_model_config,
        axial_t2_model_config
    )
    print(submission.shape)
    submission.to_csv('submission.csv', index=False)

    print(pd.read_csv('submission.csv').head().to_string())
