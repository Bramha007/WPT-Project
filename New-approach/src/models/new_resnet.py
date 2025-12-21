import torch.nn as nn
import torchvision
from src.setup.new_config_cls import NUM_CLS_CLASSES

def build_resnet_classifier(num_classes: int = NUM_CLS_CLASSES):
    """ Builds a standard ResNet-18 for 5-class Area Classification. """
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model