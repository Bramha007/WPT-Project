import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Required for headless server execution
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, LayerGradCam, visualization as viz

class XAIEngine:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def get_wrapper(self, target_idx=0):
        """Wraps output for Captum differentiability."""
        def wrapper(input_tensor):
            outputs = self.model(list(input_tensor))
            if len(outputs[0]['scores']) > target_idx:
                return outputs[0]['scores'][target_idx].view(1, 1)
            return torch.tensor([[0.0]], device=self.device, requires_grad=True)
        return wrapper

    def run_attributions(self, input_img, target_idx=0):
        """Generates Pixel (IG) and Layer-based (Grad-CAM) XAI."""
        wrapper = self.get_wrapper(target_idx)
        
        # Level 1: Pixel Attribution
        ig = IntegratedGradients(wrapper)
        attr_ig = ig.attribute(input_img, baselines=torch.ones_like(input_img), n_steps=30)
        
        # Level 2: Layer-based (Grad-CAM)
        # Targets features before they enter your custom bottleneck
        lgc = LayerGradCam(wrapper, self.model.backbone.fpn.layer_blocks[3])
        attr_lgc = lgc.attribute(input_img, target=0)
        attr_lgc = LayerGradCam.interpolate(attr_lgc, (input_img.shape[2], input_img.shape[3]))
        
        return attr_ig, attr_lgc

def save_report(img_tensor, attr_ig, attr_lgc, scores, img_id, save_dir):
    """Generates a 3-panel High-Contrast XAI Report."""
    img_np = np.transpose(img_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0))
    ig_np = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    lgc_np = attr_lgc.squeeze().cpu().detach().numpy()

    # Robust Normalization: Fixes the 'scale factor = 0' crash
    max_val = np.max(np.abs(ig_np))
    ig_np = ig_np / (max_val + 1e-9) if max_val > 0 else ig_np

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original Image + Counterfactual Proof
    axes[0].imshow(img_np)
    axes[0].set_title(f"Orig: {scores[0]:.2f} | CF Score: {scores[1]:.2f}")
    
    # Integrated Gradients Heatmap
    viz.visualize_image_attr(ig_np, img_np*0.1+0.9, method="blended_heat_map", 
                             sign="all", plt_fig_axis=(fig, axes[1]), use_pyplot=False)
    axes[1].set_title("Pixel Importance")

    # Layer Grad-CAM Heatmap
    axes[2].imshow(lgc_np, cmap='jet', alpha=0.8)
    axes[2].set_title("Layer Feature Focus")

    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, f"report_xai_{img_id}.png"))
    plt.close(fig)