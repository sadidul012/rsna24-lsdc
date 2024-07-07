import pandas as pd

from single_dataset import DATA_PATH
from single_inference import inject_series_description, instance_image_path, get_model_output

OUTPUT_DIR = "rsna24-data/models/densenet169-c3p1b16e20f14"
MODEL_NAME = "densenet169"
image_dir = DATA_PATH / f"train_images/"
dataset = pd.read_csv(DATA_PATH / "train_series_descriptions.csv")

dataset = dataset.drop_duplicates(subset=["study_id", "series_description"])
dataset = dataset.groupby("study_id").apply(lambda x: inject_series_description(x)).reset_index(drop=True)
dataset["instance_number"] = dataset.apply(lambda x: instance_image_path(x, image_dir), axis=1)
dataset = dataset.explode("instance_number")

sagittal_t2 = dataset.loc[dataset.series_description == "Sagittal T2/STIR"]
print("sagittal_t2", sagittal_t2.shape)
sagittal_t2 = get_model_output(
    sagittal_t2,
    OUTPUT_DIR + f'/sagittal_t2-best_wll_model_fold-0.pt',
    15,
    image_dir,
    MODEL_NAME
)

sagittal_t1 = dataset.loc[dataset.series_description == "Sagittal T1"]
print("sagittal_t1", sagittal_t1.shape)

sagittal_t1 = get_model_output(
    sagittal_t1,
    OUTPUT_DIR + f'/sagittal_t1-best_wll_model_fold-0.pt',
    30,
    image_dir,
    MODEL_NAME
)

axial_t2 = dataset.loc[dataset.series_description == "Axial T2"]
print("axial_t2", axial_t2.shape)

axial_t2 = get_model_output(
    axial_t2,
    OUTPUT_DIR + f'/axial_t2-best_wll_model_fold-0.pt',
    6,
    image_dir,
    MODEL_NAME
)

sagittal_t2.to_csv("rsna24-data/saggittal_t2.csv", index=False)
sagittal_t1.to_csv("rsna24-data/sagittal_t1.csv", index=False)
axial_t2.to_csv("rsna24-data/axial_t2.csv", index=False)

print("sagittal_t2", sagittal_t2.shape)
print("sagittal_t1", sagittal_t1.shape)
print("axial_t2", axial_t2.shape)
