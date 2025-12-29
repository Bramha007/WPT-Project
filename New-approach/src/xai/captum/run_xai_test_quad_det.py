import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients, visualization as viz

from src.setup import config_det as config
from src.models.fasterrcnn import build_fasterrcnn_
from src.dataio.det_dataset import GeometricShapeDataset, collate_fn
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.split_utils import subsample_pairs
from src.dataio.det_transforms import Compose, ToTensor

def run_xai_on_test_set(limit=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xai_out_dir = os.path.join(config.OUTPUT_DIR, "xai_results")
    os.makedirs(xai_out_dir, exist_ok=True)

    # 1. Load Model
    num_classes = 2
    model = build_fasterrcnn_(num_classes, config.LATENT_SIZE).to(device)
    model.load_state_dict(torch.load(config.SAVE_CKPT, map_location=device))
    model.eval()

    # 2. Prepare Data
    test_pairs = subsample_pairs(
        paired_image_xml_list(config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL_RECT), 
        config.F_TEST, seed=config.SEED
    )
    ds = GeometricShapeDataset(test_pairs, transforms=Compose([ToTensor()]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 3. Define Wrapper for Captum
    def wrapper_func(input_tensor):
        outputs = model(list(input_tensor))
        if len(outputs[0]['scores']) > 0:
            # Reshape to [1, 1] to satisfy Captum's batch requirements
            return outputs[0]['scores'][0].view(1, 1) 
        return torch.zeros((1, 1), device=device)

    ig = IntegratedGradients(wrapper_func)

    # 4. Loop and Save Explanations
    print(f"Generating XAI maps for {len(ds) if limit is None else limit} images...")
    for i, (imgs, tgts) in enumerate(tqdm(loader)):
        if limit and i >= limit: break
        
        # Prepare input with gradients enabled
        input_img = imgs[0].to(device).unsqueeze(0).requires_grad_()
        
        # FIXED: Removed target=0 and the extra call outside the loop
        attr = ig.attribute(input_img, n_steps=50, internal_batch_size=1)
        
        # Prepare for Visualization
        attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
        img_np = np.transpose(imgs[0].cpu().detach().numpy(), (1, 2, 0))

        # Create the Heatmap Overlay
        fig, ax = viz.visualize_image_attr(
            attr_np, 
            img_np, 
            method="blended_heat_map", 
            sign="all", 
            show_colorbar=True,
            use_pyplot=False  # Returns fig instead of calling plt.show()
        )
        
        # Save results
        save_path = os.path.join(xai_out_dir, f"xai_test_{i}.png")
        fig.savefig(save_path)
        plt.close(fig)

if __name__ == "__main__":
    run_xai_on_test_set(limit=1)