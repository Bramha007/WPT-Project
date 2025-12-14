import torch.nn as nn
import torchvision
from src.setup.new_config_cls import NUM_CLS_AREA, NUM_REG_WH

class MultiTaskResNet(nn.Module):
    """
    ResNet-18 with two separate heads: 
    1. Area Classification (5 classes)
    2. W/H Regression (2 continuous outputs)
    """
    def __init__(self, num_area_cls: int, num_reg_wh: int):
        super().__init__()
        
        self.backbone = torchvision.models.resnet18(weights="IMAGENET1K_V1")
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity() 

        # Task 1: Area Classification Head (5 Classes)
        self.area_head = nn.Linear(in_features, num_area_cls)
        
        # Task 2: W/H Regression Head (2 continuous outputs: W, H)
        self.wh_head = nn.Linear(in_features, num_reg_wh)

    def forward(self, x):
        features = self.backbone(x)
        area_out = self.area_head(features) # This is the classification logits
        wh_out = self.wh_head(features) # This is the regression prediction [W_pred, H_pred]
        
        return area_out, wh_out
    
def build_resnet_classifier():
    return MultiTaskResNet(NUM_CLS_AREA, NUM_REG_WH)