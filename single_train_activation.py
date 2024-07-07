import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import ast
import cv2
from torch.utils.data import Dataset

from single_dataset import process_train_csv, read_train_csv, DATA_PATH
from single_train import N_FOLDS, SEED


sagittal_t2 = pd.read_csv("rsna24-data/saggittal_t2.csv")
sagittal_t1 = pd.read_csv("rsna24-data/sagittal_t1.csv")
axial_t2 = pd.read_csv("rsna24-data/axial_t2.csv")

_train, _solution, balanced = read_train_csv(DATA_PATH)
_sagittal_t2, _sagittal_t1, _axial_t2 = process_train_csv(_train, additional_columns=True)
print(_sagittal_t2.shape, sagittal_t2.shape)


# sagittal_t2 = sagittal_t2.groupby(["study_id", "series_id", "series_description"]).apply(apply)
# sagittal_t1 = sagittal_t1.groupby(["study_id", "series_id", "series_description"]).apply(apply)
# axial_t2 = axial_t2.groupby(["study_id", "series_id", "series_description"]).apply(apply)
# df = pd.concat([sagittal_t1, axial_t2, sagittal_t2])
# # del sagittal_t1, axial_t2, sagittal_t2
#
# df = df.reset_index()
# df = df.groupby(["study_id"]).apply(lambda x: np.vstack([np.array(y).transpose((2, 0, 1)) for y in x.sort_values(by="series_description", ascending=False)[0].values]))
# df = df.reset_index()
# print(df.head())


class RSNA24DatasetActivation(Dataset):
    def __init__(self, sagittal_t2_tru, sagittal_t1_tru, axial_t2_tru, sagittal_t2_feat, sagittal_t1_feat, axial_t2_feat, split="train"):
        self.sagittal_t2_tru = sagittal_t2_tru
        self.sagittal_t1_tru = sagittal_t1_tru
        self.axial_t2_tru = axial_t2_tru
        self.sagittal_t2_feat = sagittal_t2_feat
        self.sagittal_t1_feat = sagittal_t1_feat
        self.axial_t2_feat = axial_t2_feat

        self.study_ids = np.array(sagittal_t2.study_id.unique())

        self.split = split
        self.image_size = (240, 240)
        self.label = "full_label"

    def __len__(self):
        return len(self.study_ids)

    def create_image(self, x):
        x = x.sort_values("instance_number")
        x = [ast.literal_eval(x) for x in x.preds.values]
        return cv2.resize(np.array(x), self.image_size, interpolation=cv2.INTER_CUBIC)

    def create_label(self, y, label_size):
        label = np.zeros(label_size)
        print(y)
        if y.plane.iloc[0] == "Sagittal T2/STIR" or y.plane.iloc[0] == "Sagittal T1":
            for i, x in y.iterrows():
                for full_label in x[self.label]:
                    label[full_label] = 1
        if y.plane.iloc[0] == "Axial T2":
            print(y)
        return label

    def __getitem__(self, idx):
        study_id = self.study_ids[idx]
        # y = self.create_label(self.sagittal_t2_tru.loc[self.sagittal_t2_tru.study_id == study_id], 15)
        # x = self.create_image(self.sagittal_t2_feat.loc[self.sagittal_t2_feat.study_id == study_id])
        # print(y)
        # print(x.shape)
        # y = self.create_label(self.sagittal_t1_tru.loc[self.sagittal_t1_tru.study_id == study_id], 30)
        # x = self.create_image(self.sagittal_t1_feat.loc[self.sagittal_t1_feat.study_id == study_id])
        # print(y)
        # print(x.shape)
        y = self.create_label(self.axial_t2_tru.loc[self.axial_t2_tru.study_id == study_id], 30)
        x = self.create_image(self.axial_t2_feat.loc[self.axial_t2_feat.study_id == study_id])
        print(y)
        print(x.shape)
        # self.axial_t2_tru
        # self.axial_t2_feat

        print(study_id)


ds = RSNA24DatasetActivation(_sagittal_t2, _sagittal_t1, _axial_t2, sagittal_t2, sagittal_t1, axial_t2)
print(ds.__getitem__(1))
exit()
skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
fold_score = []

for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(study_ids)))):
    print("train size", len(trn_idx), "test size", len(val_idx))

    trx_study_id = study_ids[trn_idx]
    val_study_id = study_ids[val_idx]

    df_train = df.loc[df.study_id.isin(trx_study_id)]
    df_valid = df.loc[df.study_id.isin(val_study_id)]
    print(df_train.values, df_valid.shape)
    print("train size", len(df_train), "test size", len(df_valid))

    break
