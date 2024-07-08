import glob
import os
from config import ModelConfig

models = list(glob.glob("rsna24-data/models_db/*-c3*/*.pt"))

for file in models:
    config = ModelConfig()
    config.MODEL_PATH = os.path.dirname(file)
    config.MODEL_NAME = "-".join(os.path.basename(config.MODEL_PATH).split("-")[:-1])
    config.MODEL_FILENAME = os.path.basename(file)
    config.IMG_SIZE = [512, 512]
    config.IN_CHANS = 3

    if config.MODEL_FILENAME.__contains__("sagittal_t2"):
        config.N_LABELS = 5
        config.N_CLASSES = 15

    if config.MODEL_FILENAME.__contains__("sagittal_t1"):
        config.N_LABELS = 10
        config.N_CLASSES = 30

    if config.MODEL_FILENAME.__contains__("axial_t2"):
        config.N_LABELS = 2
        config.N_CLASSES = 6

    json_file = config.MODEL_FILENAME[:-3] + ".json"
    config.save(config.MODEL_PATH + "/" + json_file)


print(len(models))
