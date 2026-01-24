import os
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg') # Required for headless server environments
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, LayerGradCam, visualization as viz

class XAIEngine:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def get_wrapper(self, target_idx=0):
        """Wraps model output for Captum differentiability."""
        def wrapper(input_tensor):
            # Faster R-CNN expects a list of tensors
            outputs = self.model(list(input_tensor))
            if len(outputs[0]['scores']) > target_idx:
                return outputs[0]['scores'][target_idx].view(1, 1)
            # Return a differentiable zero if no detection exists
            return torch.tensor([[0.0]], device=self.device, requires_grad=True)
        return wrapper

    def run_attributions(self, input_img, target_idx=0):
        """Generates both Pixel (IG) and Layer (Grad-CAM) attributions."""
        wrapper = self.get_wrapper(target_idx)
        
        # 1. Pixel Level: Integrated Gradients
        ig = IntegratedGradients(wrapper)
        attr_ig = ig.attribute(input_img, baselines=torch.ones_like(input_img), n_steps=30)
        
        # 2. Layer Level: Grad-CAM on FPN
        # We target the FPN layer blocks to see high-level feature extraction
        lgc = LayerGradCam(wrapper, self.model.backbone.fpn.layer_blocks[3])
        attr_lgc = lgc.attribute(input_img, target=0)
        attr_lgc = LayerGradCam.interpolate(attr_lgc, (input_img.shape[2], input_img.shape[3]))
        
        return attr_ig, attr_lgc

def save_report(img_tensor, attr_ig, attr_lgc, scores, img_id, save_dir):
    """Saves a 3-panel comparison plot."""
    img_np = np.transpose(img_tensor.squeeze().cpu().detach().numpy(), (1, 2, 0))
    ig_np = np.transpose(attr_ig.squeeze().cpu().detach().numpy(), (1, 2, 0))
    lgc_np = attr_lgc.squeeze().cpu().detach().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Panel 1: Original + Scores
    axes[0].imshow(img_np)
    axes[0].set_title(f"Orig Conf: {scores[0]:.2f} | CF: {scores[1]:.2f}")
    
    # Panel 2: IG Pixel Attribution
    viz.visualize_image_attr(ig_np, img_np*0.1+0.9, method="blended_heat_map", 
                             sign="all", plt_fig_axis=(fig, axes[1]), use_pyplot=False)
    axes[1].set_title("Pixel Importance (IG)")

    # Panel 3: Layer Grad-CAM
    axes[2].imshow(lgc_np, cmap='jet', alpha=0.8)
    axes[2].set_title("Bottleneck Focus (Grad-CAM)")

    plt.tight_layout()
    path = os.path.join(save_dir, f"report_xai_{img_id}.png")
    fig.savefig(path)
    plt.close(fig)
    return path