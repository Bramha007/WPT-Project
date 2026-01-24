import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Essential for the server
import matplotlib.pyplot as plt
from captum.attr import LayerGradCam

class XAILayerEngine:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def wrapper_func(self, input_tensor):
        """Standard wrapper to get the score of the top detection."""
        outputs = self.model(list(input_tensor))
        if len(outputs[0]['scores']) > 0:
            return outputs[0]['scores'][0].view(1, 1)
        return torch.tensor([[0.0]], device=self.device, requires_grad=True)

    def get_layer_attribution(self, input_img):
        """Fulfills 'layer-based XAI' requirement using Grad-CAM."""
        # We target model.backbone.fpn.layer_blocks[3] - the deepest FPN layer
        lgc = LayerGradCam(self.wrapper_func, self.model.backbone.fpn.layer_blocks[3])
        
        # Calculate attribution for the top predicted class
        attr = lgc.attribute(input_img, target=0)
        
        # Upsample the low-res Grad-CAM map to match the original image size
        attr_upsampled = LayerGradCam.interpolate(attr, (input_img.shape[2], input_img.shape[3]))
        return attr_upsampled