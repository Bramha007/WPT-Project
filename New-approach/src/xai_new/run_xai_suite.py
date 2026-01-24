import os, torch, gc
from tqdm import tqdm
from torch.utils.data import DataLoader

# Project Imports
from src.setup import config_det as config
from src.models.fasterrcnn import build_fasterrcnn_
from src.dataio.det_dataset import GeometricShapeDataset, collate_fn
from src.dataio.det_transforms import Compose, ToTensor
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.split_utils import subsample_pairs
from xai_utils import XAIEngine, save_report

def run_suite(latent_size=256, limit=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Setup Paths manually to avoid 'None' errors
    ckpt_path = f"quad_detection/latent_{latent_size}/fasterrcnn_best.pt"
    save_dir = f"quad_detection/latent_{latent_size}/xai_results"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"📂 Saving results to: {os.path.abspath(save_dir)}")

    # 2. Load Model
    model = build_fasterrcnn_(2, latent_size).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    # 3. Data Prep
    test_pairs = subsample_pairs(
        paired_image_xml_list(config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL_RECT), 
        config.F_TEST, seed=config.SEED
    )
    ds = GeometricShapeDataset(test_pairs, transforms=Compose([ToTensor()]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    engine = XAIEngine(model, device)

    for i, (imgs, _) in enumerate(tqdm(loader)):
        if i >= limit: break
        
        img_id = os.path.basename(test_pairs[i][0]).split('.')[0].replace('img_', '')
        input_img = imgs[0].to(device).unsqueeze(0).requires_grad_(True)
        
        try:
            # Run Attributions
            attr_ig, attr_lgc = engine.run_attributions(input_img)
            
            # Counterfactual Logic: Remove corner
            with torch.no_grad():
                orig_score = model([input_img.squeeze(0)])[0]['scores'][0].item()
                perturbed = input_img.clone()
                perturbed[:, :, 0:30, 0:30] = 1.0 # White out top-left corner
                cf_score = model([perturbed.squeeze(0)])[0]['scores'][0].item()

            # Save Plot
            save_report(input_img, attr_ig, attr_lgc, [orig_score, cf_score], img_id, save_dir)
            
        except Exception as e:
            print(f"Skipping ID {img_id} due to error: {e}")
        
        if i % 2 == 0: torch.cuda.empty_cache()

    print(f"\n✅ Finished! Run this on your Windows machine to pull results:")
    print(f"scp -r {os.getlogin()}@gensynth.cs.uni-magdeburg.de:{os.path.abspath(save_dir)} E:\\WPT-Project\\New-approach\\results_linux\\quad_detection\\latent_{latent_size}\\")

if __name__ == "__main__":
    # Run for the latent size you want to explain
    run_suite(latent_size=256)