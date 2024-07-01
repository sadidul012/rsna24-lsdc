import os
import time
from datetime import timedelta
import math, random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter

from dataset import RSNA24DatasetTrain, RSNA24DatasetValid
from model import RSNA24Model
from score import score


# ResNet:
#   ResNet-18: ~11.7 million parameters
#   ResNet-34: ~21.8 million parameters
#   ResNet-50: ~25.6 million parameters
#   ResNet-101: ~44.5 million parameters
#   ResNet-152: ~60 million parameters
# VGG:
#   VGG-16: ~138 million parameters
#   VGG-19: ~143 million parameters
# Inception Networks:
#   Inception v1 (GoogleNet): ~6.8 million parameters
#   Inception v3: ~23.8 million parameters
# DenseNet:
#   DenseNet-121: ~8 million parameters
#   DenseNet-169: ~14 million parameter
#   DenseNet-201: ~20 million parameters
#   DenseNet-264: ~33 million parameters - no pretrained
#   MobileNets (parameters can vary significantly with changes in alpha and resolution multipliers):
#   MobileNetV1 (1.0 224): ~4.2 million parameters
#   MobileNetV2 (1.0 224): ~3.5 million parameters
#   MobileNetV3 Large: ~5.4 million parameters
#       mobilenetv3_large_100
#   Vision Transformers (ViT):
#   ViT-B/16 (base model with patch size 16x16): ~86 million parameters
# Xception:
#   Xception: ~22.9 million parameters
# EfficientNet
#   EfficientNet-B0: ~5.3 million parameters
#   EfficientNet-B1: ~7.8 million parameters
#   EfficientNet-B2: ~9.2 million parameters
#   EfficientNet-B3: ~12 million parameters
#   EfficientNet-B4: ~19 million parameters
#   EfficientNet-B5: ~30 million parameters
#   EfficientNet-B6: ~43 million parameters
#   EfficientNet-B7: ~66 million parameters

# TODO Retrain 10 more epochs for each model
# TODO Implement competition score in test phase
# TODO Retrain 10 more epochs of the submitted models (densenet201, mobilenetv3_large_100)

# After completion of current experiments
# TODO Do not use cache for training dataset

MODEL_NAME = "timm/tf_efficientnet_b7.ra_in1k"

rd = '/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification'
DEBUG = False

PRETRAINED = True
RETRAIN = False
TRAIN = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
USE_AMP = True  # can change True if using T4 or newer than Ampere

N_WORKERS = 4
SEED = 8620
GRAD_ACC = 2
TGT_BATCH_SIZE = 8
BATCH_SIZE = TGT_BATCH_SIZE // GRAD_ACC
MAX_GRAD_NORM = None
EARLY_STOPPING_EPOCH = 5

IMG_SIZE = [512, 512]
IN_CHANS = 30
N_LABELS = 25
N_CLASSES = 3 * N_LABELS

AUG_PROB = 0.75

N_FOLDS = 5 if not DEBUG else 2
EPOCHS = 10 if not DEBUG else 2

LR = 1e-4
WD = 1e-2
AUG = True
NUMBER_OF_SAMPLES = -1 if not DEBUG else -1
OUTPUT_DIR = f'rsna24-data/rsna24-new-{MODEL_NAME}-{N_FOLDS}'

os.makedirs(OUTPUT_DIR, exist_ok=True)


def set_random_seed(seed: int = 8620, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic  # type: ignore


set_random_seed(SEED)
df = pd.read_csv(f'{rd}/train.csv')
# df = df.iloc[:100]
train_des = pd.read_csv(f'{rd}/train_series_descriptions.csv')
image_dir = f"{rd}/train_images/"

# autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.bfloat16) # if your gpu is newer Ampere, you can use this, lesser appearance of nan than half
autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)  # you can use with T4 gpu. or newer
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)
print("Using", device)

skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)


def train(model_name):
    fold_score = []
    start_time = time.time()

    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        print("train size", len(trn_idx), "test size", len(val_idx))
        df_train = df.iloc[trn_idx]
        df_valid = df.iloc[val_idx]

        train_ds = RSNA24DatasetTrain(df_train, train_des, image_dir, in_channel=IN_CHANS)
        train_dl = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            pin_memory=False,
            drop_last=True,
            num_workers=N_WORKERS
        )

        valid_ds = RSNA24DatasetValid(df_valid, train_des, image_dir, in_channel=IN_CHANS)
        valid_dl = DataLoader(
            valid_ds,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=N_WORKERS
        )
        model = RSNA24Model(model_name, IN_CHANS, N_CLASSES, pretrained=PRETRAINED)
        if RETRAIN:
            fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'
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

        weights = torch.tensor([1.0, 2.0, 4.0])
        criterion = nn.CrossEntropyLoss(weight=weights.to(device))
        criterion2 = nn.CrossEntropyLoss(weight=weights)

        best_loss = 1.2
        best_wll = 1.2
        es_step = 0
        log_dir = "rsna24-data/logs" + f"/{MODEL_NAME}/Fold-{fold}/{N_FOLDS}"

        if os.path.exists(log_dir):
            log_dir = log_dir + f"+1"

        writer = SummaryWriter(log_dir)

        for epoch in range(1, EPOCHS + 1):
            model.train()
            total_loss = 0
            with tqdm(train_dl, leave=True, desc=f"Train Fold {fold}/{N_FOLDS}, Epoch {epoch}/{EPOCHS}") as pbar:
                optimizer.zero_grad()
                for idx, (x, t) in enumerate(pbar):
                    x = x.to(device).type(torch.float32)
                    t = t.to(device)
                    with autocast:
                        loss = 0
                        y = model(x)
                        for col in range(N_LABELS):
                            pred = y[:, col * 3:col * 3 + 3]
                            gt = t[:, col]
                            loss = loss + criterion(pred, gt) / N_LABELS

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
                writer.add_scalar(f'Train/train_loss', train_loss, epoch)

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
                            for col in range(N_LABELS):
                                pred = y[:, col * 3:col * 3 + 3]
                                gt = t[:, col]

                                loss = loss + criterion(pred, gt) / N_LABELS
                                y_pred = pred.float()
                                y_preds.append(y_pred.cpu())
                                labels.append(gt.cpu())

                            total_loss += loss.item()

                val_loss = total_loss / len(valid_dl)

                y_preds = torch.cat(y_preds, dim=0)
                labels = torch.cat(labels)
                val_wll = criterion2(y_preds, labels)
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
                        fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'
                        torch.save(model.state_dict(), fname)

                    if device != 'cuda:0':
                        model.to(device)

                else:
                    es_step += 1
                    if es_step >= EARLY_STOPPING_EPOCH:
                        print('early stopping')
                        break

                print("val_loss" + f'{val_loss:.6f}', "val_wll" + f'{val_wll: .6f}')

                if old_val_loss > best_loss:
                    print("val_loss" + f'{old_val_loss: .6f}', "->" + f'{best_loss: .6f}')

                if old_wll_metric > best_wll:
                    print("wll_metric" + f'{old_wll_metric: .6f}', "->" + f'{best_wll: .6f}')

                writer.add_scalar(f'Valid/val_loss', val_loss, epoch)
                writer.add_scalar(f'Valid/val_wll', val_wll, epoch)
                writer.add_scalar(f'Valid/best_loss', best_loss, epoch)
                writer.add_scalar(f'Valid/best_wll', best_wll, epoch)

        fold_score.append(best_wll.item())
        fold_score.append(best_loss)

    fold_score.append(timedelta(seconds=time.time() - start_time))
    return fold_score


# def test_submission(model_name):
#     from inference import prepare_submission
#
#     planes = ["Sagittal T2/STIR", "Sagittal T1", "Axial T2"]
#     print(train_des.shape)
#     for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
#         df_valid = df.iloc[val_idx]
#         df_valid = df_valid.head()
#         sub = prepare_submission(train_des.loc[train_des.study_id.isin(list(df_valid.study_id))], model_name, image_dir)
#         print(sub.head())
#         print(sub.shape)
#         s = score(df_valid, sub, "row_id", 1)
#         print(s)
#         break
#
#     return 0


def test(model_name):
    y_preds = []
    labels = []
    weights = torch.tensor([1.0, 2.0, 4.0])
    criterion2 = nn.CrossEntropyLoss(weight=weights)
    fold_scores = []
    planes = ["Sagittal T2/STIR", "Sagittal T1", "Axial T2"]

    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(df)))):
        df_valid = df.iloc[val_idx]

        dataset_sagittal_t2 = RSNA24DatasetValid(df_valid, train_des, image_dir, plane="Sagittal T2/STIR", in_channel=IN_CHANS)
        dataset_sagittal_t1 = RSNA24DatasetValid(df_valid, train_des, image_dir, plane="Sagittal T1", in_channel=IN_CHANS)
        dataset_axial_t2 = RSNA24DatasetValid(df_valid, train_des, image_dir, plane="Axial T2", in_channel=IN_CHANS)

        dl_sagittal_t2 = DataLoader(dataset_sagittal_t2, batch_size=1, shuffle=False, num_workers=int(N_WORKERS/3), pin_memory=False, drop_last=False)
        dl_sagittal_t1 = DataLoader(dataset_sagittal_t1, batch_size=1, shuffle=False, num_workers=int(N_WORKERS/3), pin_memory=False, drop_last=False)
        dl_axial_t2 = DataLoader(dataset_axial_t2, batch_size=1, shuffle=False, num_workers=int(N_WORKERS/3), pin_memory=False, drop_last=False)

        model = RSNA24Model(model_name, IN_CHANS, N_CLASSES, pretrained=False)
        fname = f'{OUTPUT_DIR}/best_wll_model_fold-{fold}.pt'
        model.load_state_dict(torch.load(fname))
        model.to(device)

        model.eval()
        fold_preds = []
        fold_labels = []

        with tqdm(total=len(dataset_axial_t2), leave=True, desc=f"Test Fold {fold}") as pbar:
            with torch.no_grad():
                for idx, data in enumerate(zip(dl_sagittal_t2, dl_sagittal_t1, dl_axial_t2)):
                    (st2_x, st2_t), (st1_x, st1_t), (at2_x, at2_t) = data
                    x = torch.concatenate((st2_x, st1_x, at2_x))
                    x = x.to(device).type(torch.float32)
                    t = torch.concatenate((st2_t, st1_t, at2_t))
                    t, _ = torch.max(t, dim=0)
                    t = t.unsqueeze(0)
                    t = t.to(device)

                    with autocast:
                        y = model(x)
                        y, _ = torch.max(y, dim=0)
                        y = y.unsqueeze(0)

                        for col in range(N_LABELS):
                            pred = y[:, col * 3:col * 3 + 3]
                            gt = t[:, col]
                            y_pred = pred.float().softmax(0)
                            fold_preds.append(y_pred.cpu())
                            fold_labels.append(gt.cpu())
                            y_preds.append(y_pred.cpu())
                            labels.append(gt.cpu())

                    pbar.update()

        fold_preds = torch.cat(fold_preds)
        fold_labels = torch.cat(fold_labels)

        fold = criterion2(fold_preds, fold_labels)
        fold_scores.append(fold.item())
        print('fold score:', fold.item())

    y_preds = torch.cat(y_preds)
    labels = torch.cat(labels)
    cv = criterion2(y_preds, labels)
    print('cv score:', cv.item())
    with open(OUTPUT_DIR + f"/result", "a+") as file:
        file.write(
            f"\n\n**********\n"
            f"Folds: {N_FOLDS} Epochs: {EPOCHS} CV: {cv.item():.2f}\n"
            f"**********\n"
            f"Pretrain: {PRETRAINED}\n"
            f"Train: {TRAIN}\n"
            f"Retrain: {RETRAIN}\n"
            f"**********\n\n"
        )
    return [cv.item()] + fold_scores


if __name__ == '__main__':
    if TRAIN:
        train(MODEL_NAME)

    r = test(MODEL_NAME)
    # row = [MODEL_NAME] + r
