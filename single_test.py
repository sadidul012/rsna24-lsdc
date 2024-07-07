import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

from single_inference import prepare_submission
from score import score
from single_dataset import read_train_csv, DATA_PATH, process_train_csv
from single_train import N_FOLDS, OUTPUT_DIR, MODEL_NAME, SEED


# SEED = 42
OUTPUT_DIR = "rsna24-data/models/densenet169-c3p1b16e20f14"
MODEL_NAME = "densenet169"


def calculate_accuracy(fold_sol, sub, condition):
    spinal_canal_tru = fold_sol.loc[condition]
    spinal_canal_hat = sub.loc[condition]
    y_hat = np.argmax(spinal_canal_hat[["normal_mild", "moderate", "severe"]].values, axis=1)
    y_tru = np.argmax(spinal_canal_tru[["normal_mild", "moderate", "severe"]].values, axis=1)
    cm = confusion_matrix(y_tru, y_hat)

    return (
        f"accuracy {accuracy_score(y_tru, y_hat)} \n"
        f"normal c/w {cm[0][0] / (cm[1][0] + cm[2][0])} acc {cm[0][0] / len(y_tru[y_tru == 0])}\n"
        f"moderate c/w {cm[1][1] / (cm[0][1] + cm[2][1])} acc {cm[1][1] / len(y_tru[y_tru == 1])} \n"
        f"severe c/w {cm[2][2] / (cm[1][2] + cm[0][2])} acc {cm[2][2] / len(y_tru[y_tru == 2])}"
    )


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
        # val_study_id = val_study_id[10:20]

        fold_sol = solution.loc[solution.study_id.isin(val_study_id)].sort_values(by="row_id").reset_index(drop=True)
        spinal_canal = fold_sol.row_id.str.contains("spinal_canal")
        neural_foraminal = fold_sol.row_id.str.contains("neural_foraminal")
        subarticular = fold_sol.row_id.str.contains("subarticular")
        # print(fold_sol.to_string())
        fold_desc = train_desc.loc[train_desc.study_id.isin(val_study_id)]
        sub = prepare_submission(
            fold_desc,
            DATA_PATH / f"train_images/",
            model_location + f'/sagittal_t2-best_wll_model_fold-{fold}.pt',
            model_location + f'/sagittal_t1-best_wll_model_fold-{fold}.pt',
            model_location + f'/axial_t2-best_wll_model_fold-{fold}.pt',
            MODEL_NAME
        )
        fold_sol = fold_sol[["row_id", "normal_mild", "moderate", "severe", "sample_weight"]]

        try:
            s = score(fold_sol.copy(), sub.copy(), "row_id", 1)
            print("fold score", s)
            # print(fold_sol.head().to_string())
            # print(sub.head().to_string())
            y_hat = np.argmax(sub[["normal_mild", "moderate", "severe"]].values, axis=1)
            y_tru = np.argmax(fold_sol[["normal_mild", "moderate", "severe"]].values, axis=1)

            accuracy = accuracy_score(y_tru, y_hat)
            precision = precision_score(y_tru, y_hat, average="weighted", sample_weight=fold_sol.sample_weight)
            cm = confusion_matrix(y_tru, y_hat)
            output = f"""
Fold: {fold} Score: {s:.2f}
Accuracy {accuracy}
Precision {precision}
Confusion Matrix:
{cm}

normal correct vs wrong {cm[0][0] / (cm[1][0] + cm[2][0])}
moderate correct vs wrong {cm[1][1] / (cm[0][1] + cm[2][1])}
severe correct vs wrong {cm[2][2] / (cm[1][2] + cm[0][2])}

spinal_canal ##########
{calculate_accuracy(fold_sol, sub, spinal_canal)}

subarticular ##########
{calculate_accuracy(fold_sol, sub, subarticular)}

neural_foraminal ######
{calculate_accuracy(fold_sol, sub, neural_foraminal)}
            """
            print(output)
            with open(model_location + f"/result", "w") as file:
                file.write(output)
            scores.append(s)
        except Exception as e:
            print(e)
            print("scoring error")

        break

    print("CV:", np.mean(scores))
    # with open(model_location + f"/result", "a+") as file:
    #     file.write(
    #         f"\n\n**********\n"
    #         f"KFolds: {N_FOLDS} CV: {np.mean(scores):.2f}\n"
    #         f"\n\n***************************************\n"
    #     )


def main():
    _train, _solution, _ = read_train_csv(DATA_PATH)
    _sagittal_t2, _sagittal_t1, _axial_t2 = process_train_csv(_train)

    test(
        _train,
        _solution,
        OUTPUT_DIR
    )


if __name__ == '__main__':
    main()
