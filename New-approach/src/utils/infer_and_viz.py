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
def run_and_visualize_all(test_on_rectangles: bool = True):
    """
    Visualizes the EXACT number of images defined in the training configuration
    for the test/validation sets.
    """
    device = select_device(config_det.DEVICE)
    torch.manual_seed(config_det.SEED)
    
    latent_dim = config_det.LATENT_SIZE
    checkpoint_path = config_det.SAVE_CKPT 
    
    # 1. SETUP DATA SOURCE BASED ON DOMAIN
    tag = "rectangles" if test_on_rectangles else "squares"
    img_dir = config_det.IMG_DIR_TEST_RECT if test_on_rectangles else config_det.IMG_DIR_VAL
    xml_dir = config_det.XML_DIR_ALL_RECT if test_on_rectangles else config_det.XML_DIR_ALL

    output_viz_dir = os.path.join(config_det.OUTPUT_DIR, f"viz_{tag}_full_split")
    os.makedirs(output_viz_dir, exist_ok=True)

    # 2. MATCH TRAINING DATA FRACTION
    pairs_all = paired_image_xml_list(img_dir, xml_dir)
    # We use config_det.F_TEST to ensure the same number of images as the training script
    pairs = subsample_pairs(pairs_all, config_det.F_TEST, seed=config_det.SEED)
    
    if not pairs:
        print(f"Warning: No images found. Skipping...")
        return

    ds_val = GeometricShapeDataset(pairs, transforms=Compose([ToTensor()]))
    val_loader = DataLoader(
        ds_val, batch_size=1, shuffle=False, 
        num_workers=config_det.NUM_WORKERS, collate_fn=collate_fn
    )

    # 3. LOAD MODEL
    NUM_CLASSES = GeometricShapeDataset.get_num_classes()
    model = build_fasterrcnn_(num_classes=NUM_CLASSES, latent_dim=latent_dim).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 4. ITERATE WITHOUT LIMIT
    print(f"🚀 Visualizing ALL {len(pairs)} images for {tag} split...")
    
    for i, (imgs, tgts) in enumerate(tqdm(val_loader, desc=f"Inference: {tag}")):
        # Removed limit_count logic to process the entire split
        
        imgs = [img.to(device) for img in imgs]
        predictions_list = model(imgs)
        pred = predictions_list[0]
        
        img_tensor = imgs[0].cpu()
        target_dict = tgts[0]
        
        original_filename = os.path.basename(pairs[i][0])
        output_filename = f"pred_{original_filename.rsplit('.', 1)[0]}.png"
        output_image_path = os.path.join(output_viz_dir, output_filename)
        
        show_prediction(
            image_tensor=img_tensor,
            pred=pred,
            gt=target_dict,
            score_thr=0.7, 
            save_path=output_image_path
        )

    print(f"✅ Completed full split visualization for latent_{latent_dim}")

if __name__ == "__main__":
    pass
    # Remove limit_count to process all images defined by F_TEST in config
    # run_and_visualize_full_splits(test_on_rectangles=True)
    # run_and_visualize_full_splits(test_on_rectangles=False)