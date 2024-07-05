import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from single_inference import prepare_submission
from score import score
from single_dataset import read_train_csv, DATA_PATH, process_train_csv
from single_train import N_FOLDS, SEED, OUTPUT_DIR


def test(df, solution, model_location):
    skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_desc = pd.read_csv(DATA_PATH / "train_series_descriptions.csv")
    # print(train_desc.groupby("study_id").count().to_string())
    # exit()

    study_ids = np.array(df.study_id.unique())
    scores = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(study_ids)))):
        # if fold < 1:
        #     continue
        print(f"Test fold {fold}")
        print("train size", len(trn_idx), "test size", len(val_idx))
        # val_study_id = [3008676218, 2780132468, 2492114990]
        # val_study_id = [3008676218]
        val_study_id = study_ids[val_idx]

        fold_sol = solution.loc[solution.study_id.isin(val_study_id)].reset_index(drop=True)
        # print(fold_sol.to_string())
        fold_desc = train_desc.loc[train_desc.study_id.isin(val_study_id)]
        sub = prepare_submission(
            fold_desc,
            DATA_PATH / f"train_images/",
            model_location + f'/sagittal_t2-best_wll_model_fold-{fold}.pt',
            model_location + f'/sagittal_t1-best_wll_model_fold-{fold}.pt',
            model_location + f'/axial_t2-best_wll_model_fold-{fold}.pt'
        )

        fold_sol = fold_sol[["row_id", "normal_mild", "moderate", "severe", "sample_weight"]].sort_values(by="row_id").reset_index(drop=True)

        try:
            s = score(fold_sol, sub, "row_id", 1)
            print("fold score", s)
            with open(model_location + f"/result", "a+") as file:
                file.write(
                    f"\n\n**********\n"
                    f"Fold: {fold} Score: {s:.2f}\n"
                )
            scores.append(s)
        except Exception as e:
            print(e)
            print("scoring error")

        break

    print("CV:", np.mean(scores))
    with open(model_location + f"/result", "a+") as file:
        file.write(
            f"\n\n**********\n"
            f"KFolds: {N_FOLDS} CV: {np.mean(scores):.2f}\n"
            f"\n\n***************************************\n"
        )


if __name__ == '__main__':
    PRETRAINED = True
    _train, _solution = read_train_csv(DATA_PATH)
    _sagittal_t2, _sagittal_t1, _axial_t2 = process_train_csv(_train)

    test(
        _train,
        _solution,
        OUTPUT_DIR
    )
