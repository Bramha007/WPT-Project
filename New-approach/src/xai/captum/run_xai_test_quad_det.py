import os, torch, gc
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients, visualization as viz

# Modular Imports from your provided files
from src.setup import config_det as config
from src.utils.device_utils import select_device 
from src.models.fasterrcnn import build_fasterrcnn_
from src.dataio.det_dataset import GeometricShapeDataset, collate_fn
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.split_utils import subsample_pairs
from src.dataio.det_transforms import Compose, ToTensor

def run_proper_xai(limit=20):
    # 0. GPU Cleanup: Resolve the OutOfMemoryError
    gc.collect()
    torch.cuda.empty_cache()
    
    device = select_device(config.DEVICE)
    torch.manual_seed(config.SEED) # Match the seed from train_shapes.py
    
    xai_out_dir = os.path.join(config.OUTPUT_DIR, "xai_results")
    os.makedirs(xai_out_dir, exist_ok=True)

    # 1. Load Model with your exact latent_dim
    num_classes = 2 
    model = build_fasterrcnn_(num_classes, config.LATENT_SIZE).to(device)
    model.load_state_dict(torch.load(config.SAVE_CKPT, map_location=device))
    model.eval()

    # 2. Match the Data Split from train_shapes.py exactly
    test_pairs_all = paired_image_xml_list(config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL_RECT)
    test_pairs = subsample_pairs(test_pairs_all, config.F_TEST, seed=config.SEED)

    ds = GeometricShapeDataset(test_pairs, transforms=Compose([ToTensor()]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 3. Enhanced Wrapper: Ensure we explain high-confidence boxes
    def wrapper_func(input_tensor):
        outputs = model(list(input_tensor))
        if len(outputs[0]['scores']) > 0:
            # We explain the top-scoring box
            return outputs[0]['scores'][0].view(1, 1) 
        return torch.zeros((1, 1), device=device)

    ig = IntegratedGradients(wrapper_func)

    print(f"Generating Matched XAI for {limit} images...")
    for i, (imgs, tgts) in enumerate(tqdm(loader)):
        if i >= limit: break
        
        # FILENAME MATCHING: Extract ID to match pred_ID.png
        img_path = test_pairs[i][0]
        img_id = os.path.basename(img_path).split('.')[0].replace('img_', '')
        save_name = f"xai_{img_id}.png"
        
        input_img = imgs[0].to(device).unsqueeze(0).requires_grad_()
        
        # 4. Use a WHITE baseline (all ones) for black geometric shapes
        # This makes the "black" pixels of the quads much more significant to Captum
        white_baseline = torch.ones_like(input_img).to(device)
        
        # internal_batch_size=1 prevents the OutOfMemoryError on GPU
        attr = ig.attribute(input_img, baselines=white_baseline, n_steps=50, internal_batch_size=1)
        
        # Prepare for Visualization
        attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
        img_np = np.transpose(imgs[0].cpu().detach().numpy(), (1, 2, 0))

        # 5. Visualizer call using the suggestion from your error log
        fig, ax = viz.visualize_image_attr(
            attr_np, img_np, 
            method="blended_heat_map", 
            sign="all", 
            show_colorbar=True,
            title=f"XAI: {img_id}",
            use_pyplot=False
        )
        
        fig.savefig(os.path.join(xai_out_dir, save_name))
        plt.close(fig)
        
        # Periodic memory cleanup
        if i % 5 == 0: torch.cuda.empty_cache()

if __name__ == "__main__":
    run_proper_xai(limit=20)