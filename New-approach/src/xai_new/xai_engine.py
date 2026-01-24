import torch
import numpy as np
from captum.attr import IntegratedGradients, LayerGradCam

class XAIEngine:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def get_wrapper(self, target_idx=0):
        """Wraps model output for Captum differentiability."""
        def wrapper(input_tensor):
            outputs = self.model(list(input_tensor))
            if len(outputs[0]['scores']) > target_idx:
                # Return the score of the specific detected quad
                return outputs[0]['scores'][target_idx].view(1, 1)
            return torch.tensor([[0.0]], device=self.device, requires_grad=True)
        return wrapper

    def attribute_pixel(self, input_img, target_idx=0):
        """Pixel-level attribution using Integrated Gradients."""
        wrapper = self.get_wrapper(target_idx)
        ig = IntegratedGradients(wrapper)
        # Use white baseline for black shapes
        return ig.attribute(input_img, baselines=torch.ones_like(input_img), 
                            n_steps=50, internal_batch_size=1)

    def attribute_layer(self, input_img, target_idx=0):
        """Layer-based attribution to inspect the bottleneck features."""
        wrapper = self.get_wrapper(target_idx)
        # Target the FPN layer before your custom TwoMLPHead
        lgc = LayerGradCam(wrapper, self.model.backbone.fpn.layer_blocks[3])
        return lgc.attribute(input_img, target=0)