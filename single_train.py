import math
import os
import random
import time
from collections import OrderedDict

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from single_inference import RSNA24Model
from single_dataset import read_train_csv, DATA_PATH, process_train_csv, RSNA24DatasetBase, train_transform, balance_dataset

# TODO train and compare pretraine=true and false
# ResNet:
#   ResNet-18: ~11.7 million parameters - 0.785 (P1)
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
#   DenseNet-121: ~8 million parameters - 0.762 (P1)
#   DenseNet-161: ~26 million parameters - 0.781 (P1)
#   DenseNet-169: ~14 million parameter - 0.715 (P1) - 0.78 (S)
#   DenseNet-201: ~20 million parameters - 0.734 (P1)
#   DenseNet-264: ~33 million parameters - no pretrained
#   MobileNets (parameters can vary significantly with changes in alpha and resolution multipliers):
#   MobileNetV1 (1.0 224): ~4.2 million parameters
#   MobileNetV2 (1.0 224): ~3.5 million parameters
#   MobileNetV3 Large: ~5.4 million parameters
#       mobilenetv3_large_100
#   Vision Transformers (ViT):
#   ViT-B/16 (base model with patch size 16x16): ~86 million parameters
# Xception:
#   Xception-41: ~24.95 million parameters - 0.723 (P1) - 0.76 (S)
#   Xception-65: ~37.90 million parameters
#   Xception-71: ~40.32 million parameters
# EfficientNet
#   EfficientNet-B0: ~5.3 million parameters - 0.762 (P1)
#   EfficientNet-B1: ~7.8 million parameters - 0.758 (P1)
#   EfficientNet-B2: ~9.2 million parameters - 0.771 (P1)
#   EfficientNet-B3: ~12 million parameters - 0.770 (P1), 0.883 (P0)
#   EfficientNet-B4: ~17 million parameters - 0.759 (P1)
#   EfficientNet-B5: ~30 million parameters
#   EfficientNet-B6: ~43 million parameters
#   EfficientNet-B7: ~66 million parameters

# timm/levit_256.fb_dist_in1k - 17.88M
# timm/rexnet_150.nav_in1k - 7.84M - 0.757 (P1)
# timm/regnety_016.tv2_in1k - 10.33M - 0.784 (P1)
# timm/tinynet_e.in1k - 2M - 0.765


MODEL_NAME = "timm/tinynet_e.in1k"

rd = '/mnt/Cache/rsna-2024-lumbar-spine-degenerative-classification'
DEBUG = False

PRETRAINED = True
RETRAIN = False

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
USE_AMP = True  # can change True if using T4 or newer than Ampere

N_WORKERS = 4
SEED = 8620
GRAD_ACC = 2
TGT_BATCH_SIZE = 32
BATCH_SIZE = TGT_BATCH_SIZE // GRAD_ACC
MAX_GRAD_NORM = None
EARLY_STOPPING_EPOCH = 3

IMG_SIZE = [512, 512]
IN_CHANS = 3

AUG_PROB = 0.75
N_FOLDS = 14 if not DEBUG else 2
EPOCHS = 20 if not DEBUG else 2

LR = 1e-4
WD = 1e-2
AUG = True

MODEL_SLUG = F"DB-c{IN_CHANS}p{1 if PRETRAINED else 0}b{BATCH_SIZE}e{EPOCHS}f{N_FOLDS}"
try:
    MODEL_SLUG = f"{MODEL_NAME.split("/")[1]}-{MODEL_SLUG}"
except IndexError:
    MODEL_SLUG = f"{MODEL_NAME}-{MODEL_SLUG}"

OUTPUT_FOLDER = "rsna24-data"
OUTPUT_DIR = f'{OUTPUT_FOLDER}/{MODEL_SLUG}'
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

autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half)  # you can use with T4 gpu. or newer
scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)


def train(df, df_balanced, plane, n_classes):
    skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    n_labels = int(n_classes / 3)
    fold_score = []
    study_ids = np.array(df.study_id.unique())

    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(study_ids)))):
        trx_study_id = study_ids[trn_idx]
        val_study_id = study_ids[val_idx]

        df_train = df_balanced.loc[df_balanced.study_id.isin(trx_study_id)]
        df_valid = df.loc[df.study_id.isin(val_study_id)]
        print("train size", len(df_train), "test size", len(df_valid))

        train_ds = RSNA24DatasetBase(df_train, transform=train_transform(0.75))
        train_dl = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False, drop_last=True, num_workers=N_WORKERS
        )
        valid_ds = RSNA24DatasetBase(df_valid)
        valid_dl = DataLoader(
            valid_ds, batch_size=BATCH_SIZE * 2, shuffle=False, pin_memory=False, drop_last=False, num_workers=N_WORKERS
        )

        model = RSNA24Model(MODEL_NAME, n_classes=n_classes, pretrained=PRETRAINED)
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

    return fold_score


if __name__ == '__main__':
    from datetime import timedelta
    start = time.time()

    _train, _solution, balanced = read_train_csv(DATA_PATH)
    _sagittal_t2, _sagittal_t1, _axial_t2 = process_train_csv(_train)
    _sagittal_t2_balanced, _sagittal_t1_balanced, _axial_t2_balanced = process_train_csv(balanced)
    # print(_sagittal_t2.shape, _sagittal_t2_balanced.shape)
    # print(_sagittal_t1.shape, _sagittal_t1_balanced.shape)
    # print(_axial_t2.shape, _axial_t2_balanced.shape)
    train(_sagittal_t2, _sagittal_t2_balanced, "sagittal_t2", 15)
    train(_sagittal_t1, _sagittal_t1_balanced, "sagittal_t1", 30)
    train(_axial_t2, _axial_t2_balanced, "axial_t2", 6)

    print(f"Time taken: {timedelta(seconds=time.time() - start)}")
