# import torchvision
# from torchvision.models.detection.rpn import AnchorGenerator
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# def build_fasterrcnn_cpu(num_classes=2,
#                          anchor_sizes=((4,), (8,), (16,), (32,), (64,)),
#                          anchor_ratios=((0.5,1.0,2.0),)*5):
#     # MobileNetV3 backbone runs well on CPU
#     model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
#     model.rpn.anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
#     return model

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_fasterrcnn_cpu(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
