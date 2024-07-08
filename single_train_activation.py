import math
import os
from collections import OrderedDict

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import ast
import cv2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from single_dataset import read_train_csv, DATA_PATH
from single_inference import RSNA24Model
from single_train import N_FOLDS, SEED, set_random_seed

MODEL_NAME = "timm/rexnet_150.nav_in1k"

rd = '/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification'
DEBUG = False

PRETRAINED = True
RETRAIN = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
USE_AMP = True  # can change True if using T4 or newer than Ampere

N_WORKERS = 4
GRAD_ACC = 2
TGT_BATCH_SIZE = 32
BATCH_SIZE = TGT_BATCH_SIZE // GRAD_ACC
MAX_GRAD_NORM = None
EARLY_STOPPING_EPOCH = 10
n_classes = 75
n_labels = int(n_classes / 3)

IMG_SIZE = [240, 240]
IN_CHANS = 9

AUG_PROB = 0.75
EPOCHS = 20 if not DEBUG else 2

LR = 1e-2
WD = 1e-2
AUG = True
plane = "Activation"

MODEL_SLUG = F"A-c{IN_CHANS}p{1 if PRETRAINED else 0}b{BATCH_SIZE}e{EPOCHS}f{N_FOLDS}"
try:
    MODEL_SLUG = f"{MODEL_NAME.split("/")[1]}-{MODEL_SLUG}"
except IndexError:
    MODEL_SLUG = f"{MODEL_NAME}-{MODEL_SLUG}"

OUTPUT_FOLDER = "rsna24-data"
OUTPUT_DIR = f'{OUTPUT_FOLDER}/models/{MODEL_SLUG}'
os.makedirs(OUTPUT_DIR, exist_ok=True)

set_random_seed(SEED)

autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)  # you can use with T4 gpu. or newer
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)

sagittal_t2 = pd.read_csv("rsna24-data/saggittal_t2.csv")
sagittal_t1 = pd.read_csv("rsna24-data/sagittal_t1.csv")
axial_t2 = pd.read_csv("rsna24-data/axial_t2.csv")
train_label_df = pd.read_csv(f'{rd}/train.csv')

_train, _, _ = read_train_csv(DATA_PATH)
# _sagittal_t2, _sagittal_t1, _axial_t2 = process_train_csv(_train, additional_columns=True)
print(sagittal_t2.shape)
print(sagittal_t1.shape)
print(axial_t2.shape)


class RSNA24DatasetActivation(Dataset):
    def __init__(self, label_df, sagittal_t2_feat, sagittal_t1_feat, axial_t2_feat, st_ids, split="train"):
        self.df = label_df
        self.df = self.df.fillna(0)
        self.label2id = {'Normal/Mild': 0, 'Moderate': 1, 'Severe': 2}
        self.df = self.df.replace(self.label2id)

        self.label_df = label_df
        self.sagittal_t2_feat = sagittal_t2_feat
        self.sagittal_t1_feat = sagittal_t1_feat
        self.axial_t2_feat = axial_t2_feat

        self.study_ids = st_ids

        self.split = split
        self.image_size = (240, 240)
        self.label = "full_label"

    def __len__(self):
        return len(self.study_ids)

    def create_image(self, data):
        data = data.sort_values("instance_number")
        data = [ast.literal_eval(i) for i in data.preds.values]
        return cv2.resize(np.array(data), self.image_size, interpolation=cv2.INTER_AREA).transpose((2, 0, 1))

    def __getitem__(self, i):
        study_id = self.study_ids[i]
        xs2 = self.create_image(self.sagittal_t2_feat.loc[self.sagittal_t2_feat.study_id == study_id])
        xs1 = self.create_image(self.sagittal_t1_feat.loc[self.sagittal_t1_feat.study_id == study_id])
        xt1 = self.create_image(self.axial_t2_feat.loc[self.axial_t2_feat.study_id == study_id])
        feat = np.vstack((xs2, xs1, xt1))

        label = self.df.loc[self.df.study_id == study_id]
        label = label.values[0][1:].astype(np.int64)
        b = np.zeros((25, 3))
        b[np.arange(25), label] = 1
        label = b.reshape((75, ))
        return feat.astype(np.float32), label.astype(np.float32)


study_ids = np.array(train_label_df.study_id.unique())
# ds = RSNA24DatasetActivation(train_label_df, sagittal_t2, sagittal_t1, axial_t2, st_ids=study_ids, split="train")
# x, y = ds.__getitem__(1)

skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_score = []

for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(study_ids)))):
    trx_study_id = study_ids[trn_idx]
    val_study_id = study_ids[val_idx]

    train_ds = RSNA24DatasetActivation(
        train_label_df,
        sagittal_t2.loc[sagittal_t2.study_id.isin(trx_study_id)],
        sagittal_t1.loc[sagittal_t1.study_id.isin(trx_study_id)],
        axial_t2.loc[axial_t2.study_id.isin(trx_study_id)],
        trx_study_id,
        split="train"
    )
    train_dl = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, drop_last=True, num_workers=N_WORKERS
    )

    valid_ds = RSNA24DatasetActivation(
        train_label_df,
        sagittal_t2.loc[sagittal_t2.study_id.isin(val_study_id)],
        sagittal_t1.loc[sagittal_t1.study_id.isin(val_study_id)],
        axial_t2.loc[axial_t2.study_id.isin(val_study_id)],
        val_study_id,
        split="valid"
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=False, drop_last=False, num_workers=N_WORKERS
    )
    print("train size", len(train_ds), "test size", len(valid_ds))

    model = RSNA24Model(MODEL_NAME, in_c=IN_CHANS, n_classes=n_classes, pretrained=PRETRAINED)

    if RETRAIN:
        fname = f'{OUTPUT_DIR}/{plane}-best_wll_model_fold-{fold}.pt'
        model.load_state_dict(torch.load(fname))
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)

    warmup_steps = EPOCHS / 10 * len(train_dl) // GRAD_ACC
    num_total_steps = EPOCHS * len(train_dl) // GRAD_ACC
    num_cycles = 0.475
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_total_steps,
        num_cycles=num_cycles
    )

    weights = torch.tensor([1.0, 2.0, 4.0] * n_labels)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    criterion2 = nn.CrossEntropyLoss(weight=weights)

    best_loss = 1200
    best_wll = 1200
    es_step = 0
    log_dir = f"{OUTPUT_FOLDER}/logs" + f"/{MODEL_SLUG}/F{fold}"

    writer = SummaryWriter(log_dir)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        with tqdm(train_dl, leave=True, desc=f"Train {plane.upper()} Fold {fold}/{N_FOLDS}, Epoch {epoch}/{EPOCHS}") as pbar:
            optimizer.zero_grad()
            for idx, (x, t) in enumerate(pbar):
                x = x.to(device).type(torch.float32)
                t = t.to(device)
                with autocast:
                    loss = 0
                    y = model(x)
                    loss = loss + criterion(y, t) / n_labels
                    total_loss += loss.item()
                    if GRAD_ACC > 1:
                        loss = loss / GRAD_ACC

                if not math.isfinite(loss):
                    print(f"Loss is {loss}, stopping training")
                    continue

                pbar.set_postfix(
                    OrderedDict(
                        loss=f'{loss.item() * GRAD_ACC:.6f}',
                        lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                    )
                )
                scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM or 1e9)

                if (idx + 1) % GRAD_ACC == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()

            train_loss = total_loss / len(train_dl)
            print(f"train loss {train_loss:.6f}")
            writer.add_scalar(f'{plane.upper()}/Train/train_loss', train_loss, epoch)

        total_loss = 0
        y_preds = []
        labels = []

        model.eval()
        with tqdm(valid_dl, leave=True, desc=f"Validation") as pbar:
            with torch.no_grad():
                for idx, (x, t) in enumerate(pbar):
                    x = x.to(device).type(torch.float32)
                    t = t.to(device)
                    with autocast:
                        loss = 0
                        y = model(x)
                        loss = loss + criterion(y, t) / n_labels
                        y_preds.append(y.cpu())
                        labels.append(t.cpu())
                        total_loss += loss.item()

            val_loss = total_loss / len(valid_dl)

            y_preds = torch.cat(y_preds, dim=0)
            labels = torch.cat(labels)
            val_wll = criterion2(y_preds, labels) / n_labels
            old_val_loss = best_loss
            old_wll_metric = best_wll
            if val_loss < best_loss or val_wll < best_wll:
                es_step = 0
                if device != 'cuda:0':
                    model.to('cuda:0')

                if val_loss < best_loss:
                    # print(f'epoch:{epoch}, best loss updated from {best_loss:.6f} to {val_loss:.6f}')
                    best_loss = val_loss

                if val_wll < best_wll:
                    # print(f'epoch:{epoch}, best wll_metric updated from {best_wll:.6f} to {val_wll:.6f}')
                    best_wll = val_wll
                    fname = f'{OUTPUT_DIR}/{plane}-best_wll_model_fold-{fold}.pt'
                    torch.save(model.state_dict(), fname)

                if device != 'cuda:0':
                    model.to(device)

            else:
                es_step += 1
                if es_step >= EARLY_STOPPING_EPOCH:
                    print('early stopping')
                    break

            print("val_loss " + f'{val_loss:.6f}', "val_wll" + f'{val_wll: .6f}')

            if old_val_loss > best_loss:
                print("val_loss" + f'{old_val_loss: .6f}', "->" + f'{best_loss: .6f}')

            if old_wll_metric > best_wll:
                print("wll_metric" + f'{old_wll_metric: .6f}', "->" + f'{best_wll: .6f}')

            writer.add_scalar(f'{plane.upper()}/Valid/val_loss', val_loss, epoch)
            writer.add_scalar(f'{plane.upper()}/Valid/val_wll', val_wll, epoch)
            writer.add_scalar(f'{plane.upper()}/Valid/best_loss', best_loss, epoch)
            writer.add_scalar(f'{plane.upper()}/Valid/best_wll', best_wll, epoch)

    fold_score.append(best_wll)
    fold_score.append(best_loss)

    break
