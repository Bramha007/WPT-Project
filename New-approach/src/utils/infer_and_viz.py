import torch
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- MODULAR IMPORTS ---
from src.setup import config_det 
from src.utils.device_utils import select_device 
from src.utils.viz import show_prediction
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.det_dataset import GeometricShapeDataset, collate_fn
from src.dataio.det_transforms import Compose, ToTensor
from src.dataio.split_utils import subsample_pairs
from src.models.fasterrcnn import build_fasterrcnn_

@torch.inference_mode()
def run_and_visualize_all(test_on_rectangles: bool = True, limit_count: int | None = 10):
    """
    Directly addresses the Latent Vector variation by creating dedicated sub-folders
    under the 'quad_detection' task directory.
    """
    
    # 1. SETUP DYNAMIC PATHS (Task/Latent Folder Logic)
    device = select_device(config_det.DEVICE)
    torch.manual_seed(config_det.SEED)
    
    latent_dim = config_det.LATENT_SIZE
    checkpoint_path = config_det.SAVE_CKPT 
    
    print(f"\n--- [Visualizer] Target Task: {config_det.TASK_DIR} ---")
    print(f"--- [Visualizer] Latent Dim: {latent_dim} ---")
    
    # Determine split tag
    tag = "rectangles" if test_on_rectangles else "squares"
    img_dir = config_det.IMG_DIR_TEST_RECT if test_on_rectangles else config_det.IMG_DIR_VAL
    xml_dir = config_det.XML_DIR_ALL_RECT if test_on_rectangles else config_det.XML_DIR_ALL

    # Create visualization folder inside the latent-specific output directory
    output_viz_dir = os.path.join(config_det.OUTPUT_DIR, f"viz_{tag}")
    os.makedirs(output_viz_dir, exist_ok=True)

    # 2. PREPARE DATA LOADER
    pairs_all = paired_image_xml_list(img_dir, xml_dir)
    pairs = subsample_pairs(pairs_all, config_det.F_TEST, seed=config_det.SEED)
    
    if not pairs:
        print(f"Warning: No images found in {img_dir}. Skipping...")
        return

    ds_val = GeometricShapeDataset(pairs, transforms=Compose([ToTensor()]))
    val_loader = DataLoader(
        ds_val, batch_size=1, shuffle=False, 
        num_workers=config_det.NUM_WORKERS, collate_fn=collate_fn
    )

    # 3. LOAD THE MODEL WITH CORRECT LATENT DIM
    NUM_CLASSES = GeometricShapeDataset.get_num_classes()
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading weights from: {checkpoint_path}")
    model = build_fasterrcnn_(num_classes=NUM_CLASSES, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 4. ITERATE AND PREDICT
    print(f"Saving {tag} visualizations to: {output_viz_dir}")
    
    for i, (imgs, tgts) in enumerate(tqdm(val_loader, desc=f"Inference: {tag}")):
        if limit_count is not None and i >= limit_count: break
        
        imgs = [img.to(device) for img in imgs]
        predictions_list = model(imgs)
        pred = predictions_list[0]
        
        # --- [NEW] PREPARE SCORES FOR DISPLAY ---
        scores = pred['scores'].cpu().numpy()
        img_id = os.path.basename(pairs[i][0]).split('.')[0]
        
        # Format scores to be printed on the images
        # This explains why the right-most shape in ID 5109 might be 'hidden'
        custom_labels = [f"Conf: {s:.3f}" for s in scores]

        # --- [NEW] COUNTERFACTUAL CHECK ---
        if len(scores) > 0:
            with torch.no_grad():
                perturbed_img = imgs[0].clone()
                # Remove a corner to see if geometry matters
                perturbed_img[:, 0:40, 0:40] = 1.0 
                cf_preds = model([perturbed_img])
                cf_score = cf_preds[0]['scores'][0].item() if len(cf_preds[0]['scores']) > 0 else 0.0
                print(f"ID {img_id} | Orig Score: {scores[0]:.3f} | CF Score: {cf_score:.3f} | Drop: {scores[0]-cf_score:.3f}")

        img_tensor = imgs[0].cpu()
        target_dict = tgts[0]
        
        output_filename = f"pred_{img_id}.png"
        output_image_path = os.path.join(output_viz_dir, output_filename)
        
        # Pass score_labels to print the confidence values directly on the image
        show_prediction(
            image_tensor=img_tensor,
            pred=pred,
            gt=target_dict,
            score_thr=0.7, 
            labels=custom_labels, 
            save_path=output_image_path
        )

    print(f"✅ Completed visualizations and score analysis for latent_{latent_dim}")

if __name__ == "__main__":
    # Test on Rectangles (Task Domain Shift)
    run_and_visualize_all(test_on_rectangles=True, limit_count=20)
    # Test on Squares (Training Distribution)
    run_and_visualize_all(test_on_rectangles=False, limit_count=20)