import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients, visualization as viz
from src.setup import config_det as config
from src.models.fasterrcnn import build_flexible_fasterrcnn

def explain_prediction(image_tensor, model, target_class=1):
    """
    Generates a heatmap showing pixel attribution for a specific detection.
    """
    model.eval()
    
    # We wrap the model to return only the score for the target class
    def model_forward(input_batch):
        # Faster R-CNN expects a list of tensors
        outputs = model(list(input_batch))
        # Return the max score for the target_class across all detected boxes
        scores = [out['scores'][out['labels'] == target_class] for out in outputs]
        # Handle cases with no detections
        return torch.stack([s[0] if len(s) > 0 else torch.tensor(0.0).to(s.device) for s in scores])

    # Initialize Integrated Gradients
    ig = IntegratedGradients(model_forward)
    
    # Input must have gradients enabled for Captum
    input_img = image_tensor.unsqueeze(0).requires_grad_()
    
    # Calculate attribution scores
    attributions = ig.attribute(input_img, target=0) # Target 0 of the model_forward output
    
    # Transpose for visualization [H, W, C]
    attributions_np = np.transpose(attributions.squeeze().cpu().detach().numpy(), (1, 2, 0))
    img_np = np.transpose(image_tensor.cpu().detach().numpy(), (1, 2, 0))

    # Visualize side-by-side
    fig, ax = viz.visualize_image_attr_overlay(
        attributions_np, 
        img_np, 
        method="blended_heat_map", 
        sign="all", 
        show_colorbar=True,
        title="Pixel Attribution for Quad Detection"
    )
    return fig

# Usage Example
# fig = explain_prediction(my_image_tensor, my_trained_model)
# fig.savefig(os.path.join(config.OUTPUT_DIR, "xai_explanation.png"))