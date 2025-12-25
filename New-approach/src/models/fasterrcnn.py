# # import torchvision
# # from torchvision.models.detection.rpn import AnchorGenerator
# # from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# # def build_fasterrcnn_cpu(num_classes=2,
# #                          anchor_sizes=((4,), (8,), (16,), (32,), (64,)),
# #                          anchor_ratios=((0.5,1.0,2.0),)*5):
# #     # MobileNetV3 backbone runs well on CPU
# #     model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
# #     model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
# #     in_features = model.roi_heads.box_predictor.cls_score.in_features
# #     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# #     return model

# # import torchvision
# # from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# # def build_fasterrcnn_cpu(num_classes=2):
# #     model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
# #     in_features = model.roi_heads.box_predictor.cls_score.in_features
# #     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
# #     return model


# # File: src/models/fasterrcnn.py (Renamed and Modularized)

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from typing import Tuple, List

# def build_fasterrcnn(
#     num_classes: int,
#     backbone_weights: str = "DEFAULT",
#     # anchor_sizes: Tuple[Tuple[int, ...], ...] | None = None,
#     # anchor_ratios: Tuple[Tuple[float, ...], ...] | None = None
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

#     # --- RPN Anchor Customization (Optional for small objects) ---
#     # if anchor_sizes and anchor_ratios:
#     #     print("Note: Custom Anchor Generator applied.")
#     #     model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
    
#     # --- Classification Head Replacement ---
#     # Get the number of input features for the box predictor
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
    
#     # Replace the existing head with a new one for our num_classes
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
#     return model

def build_fasterrcnn(num_classes):
    """
    Reverted to the Standard Faster R-CNN architecture.
    Uses pre-trained 1024-dim representation for stable results.
    """
    # Load the standard model with FPN backbone
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    # The default 'in_features' for this head is 1024
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Only replace the final predictor to match Background + Shape
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# # File: src/models/fasterrcnn.py (Simplified and Robust)

# import torchvision
# from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from typing import Tuple, List

# def build_fasterrcnn(
#     num_classes: int,
#     backbone_weights: str = "DEFAULT"
#     # Removed anchor_sizes and anchor_ratios parameters
# ):
#     """
#     Builds a Faster R-CNN model (MobileNetV3 FPN) and enforces small anchors
#     by overriding the RPN's default sizes.
#     """
    
#     # 1. Load the pre-trained model with default anchors
#     model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
#         weights=backbone_weights if backbone_weights else None
#     )

#     # 2. Extract current sizes/ratios from the default anchor generator
#     default_anchor_sizes = model.rpn.anchor_generator.sizes
#     default_anchor_ratios = model.rpn.anchor_generator.aspect_ratios
    
#     # 3. Define the small sizes needed for your quadrilaterals (4, 8, 16, 32)
#     new_sizes = (4, 8, 16, 32) 
    
#     # CRITICAL FIX: Replace the existing anchor sizes with a tuple that includes
#     # your new small sizes, while keeping the number of outer tuples the same.
#     # We assume the default has 5 layers (P2-P6) and replace the first few with your small sizes.
#     # We MUST maintain the number of outer tuples (5 or 6) that the default RPN setup expects.
    
#     new_anchor_sizes = list(default_anchor_sizes) # Convert tuple of tuples to list
    
#     # Overwrite the first three layers with your desired small sizes (4, 8, 16)
#     # This is a safe way to introduce small anchors without changing the layer count.
#     new_anchor_sizes[0] = (4,) 
#     new_anchor_sizes[1] = (8,)
#     new_anchor_sizes[2] = (16,)
    
#     # 4. Re-instantiate the AnchorGenerator with the modified sizes
#     model.rpn.anchor_generator = AnchorGenerator(
#         tuple(new_anchor_sizes), # Convert back to Tuple[Tuple]
#         default_anchor_ratios
#     )
    
#     # --- Classification Head Replacement ---
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
#     return model



import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead

def build_flexible_fasterrcnn(num_classes, latent_dim=None):
    """
    Capability to switch between Standard and Bottleneck architectures.
    
    Args:
        num_classes: Number of output classes (e.g., Background + Shape).
        latent_dim: If int, replaces internal layers with a custom bottleneck.
                    If None, uses the standard 1024-dim pre-trained architecture.
    """
    # Load the base model with pre-trained weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")

    if latent_dim is not None:
        # --- BOTTLENECK MODE (Custom Variation) ---
        # 1. Identify input features from the backbone (Default 12544)
        in_channels = model.roi_heads.box_head.fc6.in_features 

        # 2. Replace Box Head with custom dimension (Randomly Initialized)
        model.roi_heads.box_head = TwoMLPHead(in_channels, latent_dim)

        # 3. Replace Predictor to match the custom latent_dim
        model.roi_heads.box_predictor = FastRCNNPredictor(latent_dim, num_classes)
        print(f"Model built with CUSTOM BOTTLENECK: {latent_dim} units")
    else:
        # --- STANDARD MODE (Baseline) ---
        # 1. Keep the pre-trained 1024-dim internal architecture
        in_features = model.roi_heads.box_predictor.cls_score.in_features

        # 2. Only replace the final output predictor
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        print("Model built with STANDARD 1024-dim architecture (Pre-trained)")

    return model