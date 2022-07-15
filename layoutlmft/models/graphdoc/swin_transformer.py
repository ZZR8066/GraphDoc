import torch
from torch import nn
from mmdet.models import build_backbone, build_neck


class VisionBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = build_backbone(config.backbone_cfg)
        self.neck = build_neck(config.neck_cfg)
        self.freeze = config.vision_freeze

    def forward(self, img):
        """Directly extract features from the backbone+neck."""
        if self.freeze:
            with torch.no_grad():
                x = self.backbone(img)
                x = self.neck(x)
        else:
            x = self.backbone(img)
            x = self.neck(x)
        return x