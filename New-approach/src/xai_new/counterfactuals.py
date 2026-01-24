import torch

def generate_geometric_counterfactual(model, img, device, box_idx=0):
    """Simulates a counterfactual by removing a vertex area."""
    model.eval()
    with torch.no_grad():
        # Get baseline confidence
        orig_out = model([img.squeeze(0).to(device)])[0]
        orig_score = orig_out['scores'][box_idx].item() if len(orig_out['scores']) > box_idx else 0
        
        # Perturbation: White out the top-left corner of the detected box
        perturbed_img = img.clone()
        box = orig_out['boxes'][box_idx].int()
        # 'Remove' the corner by making it white (background color)
        perturbed_img[:, :, box[1]:box[1]+15, box[0]:box[0]+15] = 1.0
        
        cf_out = model([perturbed_img.squeeze(0).to(device)])[0]
        cf_score = cf_out['scores'][box_idx].item() if len(cf_out['scores']) > box_idx else 0
        
    return orig_score, cf_score, perturbed_img