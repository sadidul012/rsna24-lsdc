import timm
from torch import nn


class RSNA24Model(nn.Module):
    def __init__(self, model_name, in_c=3, n_classes=30, pretrained=True, features_only=False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=features_only,
            in_chans=in_c,
            num_classes=n_classes,
            global_pool='avg'
        )

    def forward(self, x):
        y = self.model(x)
        return y
