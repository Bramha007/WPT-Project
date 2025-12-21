# import torchvision
# from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from typing import Tuple, List

# def build_fasterrcnn(
#     num_classes: int,
#     backbone_weights: str = "DEFAULT",
# ):
#     """
#     Builds a Faster R-CNN model with MobileNetV3 backbone.
#     The model is device-agnostic; device selection happens in the training script.
#     """
#     # Load the pre-trained MobileNetV3 + FPN model
#     # Weights are used for transfer learning unless set to None

#     model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
#         weights=backbone_weights if backbone_weights else None
#     )


#     # --- Classification Head Replacement ---
#     # Get the number of input features for the box predictor
#     in_features = model.roi_heads.box_predictor.cls_score.in_features

#     # Replace the existing head with a new one for our num_classes
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#     return model


import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead

def build_fasterrcnn(num_classes, latent_dim=1024):
    """
    Builds Faster R-CNN with a customizable latent vector (representation_size).
    """
    # Load model with FPN backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # 1. Identify input features from the backbone
    # Default for ResNet50-FPN is 12544 (7x7 x 256)
    in_channels = model.roi_heads.box_head.fc6.in_features 

    # 2. VARIATION: Replace the Box Head with custom latent_dim
    # This 'representation_size' is your latent vector size.
    model.roi_heads.box_head = TwoMLPHead(in_channels, latent_dim)

    # 3. Replace the predictor to match the new latent_dim
    model.roi_heads.box_predictor = FastRCNNPredictor(latent_dim, num_classes)

    return model
