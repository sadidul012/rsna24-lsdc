import json


class ModelConfig:
    MODEL_PATH = 'rsna24-data/models_db/xception41-DB-c3p1b16e20f14/'
    MODEL_FILENAME = 'axial_t2-best_wll_model_fold-0.pt'
    MODEL_NAME = "xception41"

    IMG_SIZE = [512, 512]
    IN_CHANS = 3

    N_LABELS = 10
    N_CLASSES = 30

    def __init__(self, path=None):
        if path is not None:
            self.load(path)

    def load(self, path):
        with open(path, 'r') as f:
            config = json.load(f)
            self.__dict__.update(config)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)


if __name__ == '__main__':
    c = ModelConfig()
    c.load('config.json')
    print(c.MODEL)
    print(c.IMAGE_SIZE)
