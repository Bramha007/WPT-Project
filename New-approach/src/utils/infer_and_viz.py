import torch
import os
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
    
    # 1. SETUP DYNAMIC PATHS
    device = select_device(config_det.DEVICE)
    torch.manual_seed(config_det.SEED)
    
    latent_dim = config_det.LATENT_SIZE
    checkpoint_path = config_det.SAVE_CKPT 
    
    print(f"\n--- [Visualizer] Target Task: {config_det.TASK_DIR} ---")
    print(f"--- [Visualizer] Latent Dim: {latent_dim} ---")
    
    tag = "rectangles" if test_on_rectangles else "squares"
    img_dir = config_det.IMG_DIR_TEST_RECT if test_on_rectangles else config_det.IMG_DIR_VAL
    xml_dir = config_det.XML_DIR_ALL_RECT if test_on_rectangles else config_det.XML_DIR_ALL

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

    # 3. LOAD THE MODEL
    NUM_CLASSES = GeometricShapeDataset.get_num_classes()
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return

    print(f"Loading weights from: {checkpoint_path}")
    model = build_fasterrcnn_(num_classes=NUM_CLASSES, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 4. ITERATE, PREDICT, AND PRINT SCORES
    print(f"Saving {tag} visualizations to: {output_viz_dir}")
    
    for i, (imgs, tgts) in enumerate(tqdm(val_loader, desc=f"Inference: {tag}")):
        if limit_count is not None and i >= limit_count: break
        
        imgs = [img.to(device) for img in imgs]
        predictions_list = model(imgs)
        pred = predictions_list[0]
        
        # --- NEW: SCORE PRINTING LOGIC ---
        # Extracts raw scores to explain 'hidden' detections seen in XAI
        scores = pred['scores'].cpu().numpy()
        img_id = os.path.basename(pairs[i][0]).split('.')[0]
        
        if len(scores) > 0:
            # Print the top detection score to the console
            print(f"ID {img_id} | Highest Confidence Score: {scores[0]:.4f}")
            
            # Identify if the top score is below your threshold
            if scores[0] < 0.7:
                print(f"   ⚠️  Note: This object exists but is HIDDEN in the image (Score < 0.7)")
        else:
            print(f"ID {img_id} | No objects detected by the model")

        img_tensor = imgs[0].cpu()
        target_dict = tgts[0]
        
        output_filename = f"pred_{img_id}.png"
        output_image_path = os.path.join(output_viz_dir, output_filename)
        
        # Original visualization with the 0.7 threshold
        show_prediction(
            image_tensor=img_tensor,
            pred=pred,
            gt=target_dict,
            score_thr=0.7, 
            save_path=output_image_path
        )

    print(f"✅ Completed visualizations and score analysis for latent_{latent_dim}")

if __name__ == "__main__":
    run_and_visualize_all(test_on_rectangles=True, limit_count=20)
    run_and_visualize_all(test_on_rectangles=False, limit_count=20)