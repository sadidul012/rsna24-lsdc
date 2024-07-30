import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score

from config import ModelConfig
from single_inference import prepare_submission
from score import score
from single_dataset import read_train_csv, DATA_PATH, process_train_csv
from single_train import N_FOLDS, SEED, model_config

MODEL_PATH = model_config.MODEL_PATH
# MODEL_PATH = "/home/sadid-dl/PycharmProjects/rsna24-lsdc/rsna24-data/models/tinynet_e.in1k-c3p1b16e20f14"
# MODEL_PATH = "/home/sadid-dl/PycharmProjects/rsna24-lsdc/rsna24-data/models_db/densenet169-DB-c3p1b16e20f14"
# sagittal_t2_model_config = ModelConfig(MODEL_PATH + "/sagittal_t2-best_wll_model_fold-0.json")
# sagittal_t1_model_config = ModelConfig(MODEL_PATH + "/sagittal_t1-best_wll_model_fold-0.json")
# axial_t2_model_config = ModelConfig(MODEL_PATH + "/axial_t2-best_wll_model_fold-0.json")
sagittal_t2_model_config = ModelConfig("/home/sadid-dl/PycharmProjects/rsna24-lsdc/rsna24-data/models/densenet161-c3p1b16e20f14/sagittal_t2-best_wll_model_fold-0.json")
sagittal_t1_model_config = ModelConfig("/home/sadid-dl/PycharmProjects/rsna24-lsdc/rsna24-data/models_db/xception41-DB-c3p1b16e20f14/sagittal_t1-best_wll_model_fold-0.json")
axial_t2_model_config = ModelConfig("/home/sadid-dl/PycharmProjects/rsna24-lsdc/rsna24-data/models_db/xception41-DB-c3p1b16e20f14/axial_t2-best_wll_model_fold-0.json")

activation_model_config = ModelConfig("/home/sadid-dl/PycharmProjects/rsna24-lsdc/rsna24-data/models/rexnet_150.nav_in1k-A-c9p1b16e20f14/Activation-best_wll_model_fold-0.json")

# RESULT_DIRECTORY = activation_model_config.MODEL_PATH
RESULT_DIRECTORY = None
# RESULT_DIRECTORY = MODEL_PATH
print("result dir", RESULT_DIRECTORY)
METHOD = "average"
# METHOD = "activation"


def calculate_condition_metrics(fold_sol, sub, condition=None, return_cm=False):
    y_hat = sub.copy().sort_values("row_id").reset_index(drop=True)
    y_tru = fold_sol.copy().sort_values("row_id").reset_index(drop=True)
    if condition is not None:
        y_tru = fold_sol.loc[fold_sol.row_id.str.contains(condition)]
        y_hat = sub.loc[sub.row_id.str.contains(condition)]

    y_hat = np.argmax(y_hat[["normal_mild", "moderate", "severe"]].values, axis=1)
    y_tru = np.argmax(y_tru[["normal_mild", "moderate", "severe"]].values, axis=1)
    cm = confusion_matrix(y_tru, y_hat)

    if return_cm:
        return accuracy_score(y_tru, y_hat), cm

    return (
        f"accuracy {accuracy_score(y_tru, y_hat)} \n"
        f"normal   c/w {cm[0][0] / (cm[1][0] + cm[2][0]):.3f} acc {cm[0][0] / len(y_tru[y_tru == 0]):.3f}\n"
        f"moderate c/w {cm[1][1] / (cm[0][1] + cm[2][1]):.3f} acc {cm[1][1] / len(y_tru[y_tru == 1]):.3f} \n"
        f"severe   c/w {cm[2][2] / (cm[1][2] + cm[0][2]):.3f} acc {cm[2][2] / len(y_tru[y_tru == 2]):.3f}"
    )


def get_sub(val_study_id):
    sub = pd.read_csv("/home/sadid-dl/PycharmProjects/RSNA-2024-Lumbar-Spine-Degenerative-Classification/submission.csv")
    sub["study_id"] = sub.row_id.apply(lambda x: int(x.split("_")[0]))
    sub = sub.loc[sub.study_id.isin(val_study_id)].reset_index(drop=True)[["row_id", "normal_mild", "moderate", "severe"]]
    return sub


def calculate_scores(sub, fold_sol):
    sub = sub.copy().sort_values("row_id").reset_index(drop=True)
    fold_sol = fold_sol.copy().sort_values("row_id").reset_index(drop=True)

    fold_sol = fold_sol[["row_id", "normal_mild", "moderate", "severe", "sample_weight"]]
    s = score(fold_sol.copy(), sub.copy(), "row_id", 1)
    y_hat = np.argmax(sub[["normal_mild", "moderate", "severe"]].values, axis=1)
    y_tru = np.argmax(fold_sol[["normal_mild", "moderate", "severe"]].values, axis=1)

    accuracy = accuracy_score(y_tru, y_hat)
    precision = precision_score(y_tru, y_hat, average="weighted", sample_weight=fold_sol.sample_weight)
    cm = confusion_matrix(y_tru, y_hat)

    return accuracy, precision, cm, s


def test(df, solution):
    skf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_desc = pd.read_csv(DATA_PATH / "train_series_descriptions.csv")
    # print(train_desc.groupby("study_id").count().to_string())
    # exit()

    study_ids = np.array(df.study_id.unique())
    scores = []
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(study_ids)))):
        if fold != 0:
            continue
        print(f"Test fold {fold}")
        print("train size", len(trn_idx), "test size", len(val_idx))
        # val_study_id = [3008676218, 2780132468, 2492114990]  # less folder
        # val_study_id = [3008676218]  # less folder
        # val_study_id = [22191399]  # good
        val_study_id = study_ids[val_idx]
        # val_study_id = val_study_id[10:20]

        fold_sol = solution.loc[solution.study_id.isin(val_study_id)].sort_values(by="row_id").reset_index(drop=True)
        # print(fold_sol.to_string())
        fold_desc = train_desc.loc[train_desc.study_id.isin(val_study_id)]
        sub = prepare_submission(
            fold_desc,
            DATA_PATH / f"train_images/",
            activation_model_config,
            sagittal_t2_model_config,
            sagittal_t1_model_config,
            axial_t2_model_config,
            method=METHOD
        )
        if RESULT_DIRECTORY is not None:
            sub.to_csv(RESULT_DIRECTORY + '/submission.csv', index=False)
        # sub = get_sub(val_study_id)
        try:
            accuracy, precision, cm, s = calculate_scores(sub, fold_sol)
            print("fold score", s)

            output = f"""
####################################
Fold: {fold} Score: {s:.2f}
Accuracy {accuracy:.3f}
Precision {precision:.3f}
Confusion Matrix:
{cm}

normal correct vs wrong {cm[0][0] / (cm[1][0] + cm[2][0]):.3f}
moderate correct vs wrong {cm[1][1] / (cm[0][1] + cm[2][1]):.3f}
severe correct vs wrong {cm[2][2] / (cm[1][2] + cm[0][2]):.3f}

spinal_canal ##########
{calculate_condition_metrics(fold_sol, sub, "spinal_canal")}

subarticular ##########
{calculate_condition_metrics(fold_sol, sub, "subarticular")}

neural_foraminal ######
{calculate_condition_metrics(fold_sol, sub, "neural_foraminal")}
            """
            print(output)
            if RESULT_DIRECTORY is not None:
                with open(RESULT_DIRECTORY + f"/result", "w") as file:
                    file.write(output)
            scores.append(s)
        except Exception as e:
            print(e)
            print("scoring error")

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
        _solution
    )


if __name__ == '__main__':
    main()
