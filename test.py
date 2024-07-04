import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from single_inference import prepare_submission
from score import score
from single_dataset import read_train_csv, DATA_PATH, process_train_csv
from train import N_FOLDS, SEED


def test(df, solution):
    skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_desc = pd.read_csv(DATA_PATH / "train_series_descriptions.csv")
    # print(train_desc.groupby("study_id").count().to_string())
    # exit()

    study_ids = np.array(df.study_id.unique())
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(study_ids)))):
        print(f"Test fold {fold}")
        print("train size", len(trn_idx), "test size", len(val_idx))
        # val_study_id = [3008676218, 2780132468, 2492114990]
        # val_study_id = [3008676218]
        val_study_id = study_ids[val_idx]

        fold_sol = solution.loc[solution.study_id.isin(val_study_id)].reset_index(drop=True)
        fold_desc = train_desc.loc[train_desc.study_id.isin(val_study_id)]

        sub = prepare_submission(fold_desc, DATA_PATH / f"train_images/")
        s = score(fold_sol[["row_id", "normal_mild", "moderate", "severe", "sample_weight"]], sub, "row_id", 1)
        print("score", s)
        break


if __name__ == '__main__':
    PRETRAINED = True
    _train, _solution = read_train_csv(DATA_PATH)
    _sagittal_t2, _sagittal_t1, _axial_t2 = process_train_csv(_train)

    test(_train, _solution)
