import torchvision
import torch.nn as nn

def build_resnet_classifier(num_classes=5):
    model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
