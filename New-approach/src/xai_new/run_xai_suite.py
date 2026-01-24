import os, torch, gc
import matplotlib.pyplot as plt
from tqdm import tqdm
from captum.attr import visualization as viz
from src.xai_new.xai_engine import XAIEngine
from src.xai_new.counterfactuals import generate_geometric_counterfactual
from src.setup import config_det as config
from src.models.fasterrcnn import build_fasterrcnn_
from src.dataio.det_dataset import GeometricShapeDataset, collate_fn
from src.dataio.det_transforms import Compose, ToTensor
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.split_utils import subsample_pairs

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_fasterrcnn_(2, config.LATENT_SIZE).to(device)
    model.load_state_dict(torch.load(config.SAVE_CKPT, map_location=device))
    
    engine = XAIEngine(model, device)
    test_pairs = subsample_pairs(paired_image_xml_list(config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL_RECT), 
                                 config.F_TEST, seed=config.SEED)
    
    for i in tqdm(range(10)): # Run on first 10 test images
        img, _ = GeometricShapeDataset([test_pairs[i]], transforms=Compose([ToTensor()]))[0]
        input_img = img.unsqueeze(0).to(device).requires_grad_(True)
        
        # 1. Pixel Attribution
        attr_ig = engine.attribute_pixel(input_img)
        # 2. Counterfactual check
        orig_s, cf_s, _ = generate_geometric_counterfactual(model, img.unsqueeze(0), device)
        
        # Visual Summary
        print(f"ID {i} | Conf: {orig_s:.2f} | CF (Corner Removed): {cf_s:.2f}")
        
        # Save visualization logic here using viz.visualize_image_attr...
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()