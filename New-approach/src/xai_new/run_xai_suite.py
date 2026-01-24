import os, torch, gc
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.setup import config_det as config
from src.models.fasterrcnn import build_fasterrcnn_
from src.dataio.det_dataset import GeometricShapeDataset, collate_fn
from src.dataio.det_transforms import Compose, ToTensor
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.split_utils import subsample_pairs
from xai_utils import XAIEngine, save_report

def run_suite(latent_size=256, limit=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Path setup manually to prevent latent_None errors
    ckpt_path = f"quad_detection/latent_{latent_size}/fasterrcnn_best.pt"
    save_dir = f"quad_detection/latent_{latent_size}/xai_report"
    
    # Verify write access
    os.makedirs(save_dir, exist_ok=True)
    
    model = build_fasterrcnn_(2, latent_size).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    test_pairs = subsample_pairs(paired_image_xml_list(config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL_RECT), 
                                 config.F_TEST, seed=config.SEED)
    loader = DataLoader(GeometricShapeDataset(test_pairs, Compose([ToTensor()])), 1, False, collate_fn=collate_fn)

    engine = XAIEngine(model, device)
    print(f"✅ Executing XAI... Writing to: {os.path.abspath(save_dir)}")

    for i, (imgs, _) in enumerate(tqdm(loader)):
        if i >= limit: break
        img_id = os.path.basename(test_pairs[i][0]).split('.')[0].replace('img_', '')
        input_img = imgs[0].to(device).unsqueeze(0).requires_grad_(True)
        
        try:
            # Generate Attributions
            attr_ig, attr_lgc = engine.run_attributions(input_img)
            
            # Level 3: Counterfactual Proof (Removing top-left corner)
            with torch.no_grad():
                res = model([input_img.squeeze(0)])[0]
                orig_score = res['scores'][0].item() if len(res['scores']) > 0 else 0
                perturbed = input_img.clone()
                perturbed[:, :, 0:30, 0:30] = 1.0 # White out corner
                cf_score = model([perturbed.squeeze(0)])[0]['scores'][0].item() if len(res['scores']) > 0 else 0

            save_report(input_img, attr_ig, attr_lgc, [orig_score, cf_score], img_id, save_dir)
            
        except Exception as e:
            print(f"Error on ID {img_id}: {e}")
        
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Specify the latent size for the run
    run_suite(latent_size=256)