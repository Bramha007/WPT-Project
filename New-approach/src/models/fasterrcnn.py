import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead

def build_fasterrcnn_(num_classes, latent_dim=None):
    """
    Modified version of your first function to support latent vectors.
    """   
    if latent_dim is not None:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
        # --- BOTTLENECK MODE ---
        # Get input channels from the MobileNet backbone (usually 960 for V3-Large)
        in_channels = model.roi_heads.box_head.fc6.in_features 
        # Replace internal head with custom dimension (This fulfills the project requirement)
        model.roi_heads.box_head = TwoMLPHead(in_channels, latent_dim)
        # Replace predictor to match the new latent_dim
        model.roi_heads.box_predictor = FastRCNNPredictor(latent_dim, num_classes)
        print(f"Built MobileNetV3 with CUSTOM LATENT: {latent_dim}")
    else:
        # --- STANDARD MODE (Baseline) ---
        # Keep the native 1024-dim pre-trained representation
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        print("Built MobileNetV3 with STANDARD architecture")

    return model

