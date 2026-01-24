import os, torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.setup import config_det as config
from src.models.fasterrcnn import build_fasterrcnn_
from src.dataio.det_dataset import GeometricShapeDataset, collate_fn
from src.dataio.det_transforms import Compose, ToTensor
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.split_utils import subsample_pairs
from src.xai_new.xai_utils import XAILayerEngine
import matplotlib.pyplot as plt

def main():
    latent_size = 256 # Start with 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Simple pathing
    ckpt = f"quad_detection/latent_{latent_size}/fasterrcnn_best.pt"
    out_dir = f"quad_detection/latent_{latent_size}/layer_xai_only"
    os.makedirs(out_dir, exist_ok=True)

    model = build_fasterrcnn_(2, latent_size).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    
    engine = XAILayerEngine(model, device)
    
    # Load test data
    test_pairs = subsample_pairs(paired_image_xml_list(config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL_RECT), 
                                 config.F_TEST, seed=config.SEED) 
    loader = DataLoader(GeometricShapeDataset(test_pairs, Compose([ToTensor()])), 1, collate_fn=collate_fn)

    print(f"Generating Layer XAI in: {out_dir}")
    for i, (imgs, _) in enumerate(tqdm(loader)):
        
        img_id = os.path.basename(test_pairs[i][0]).split('.')[0]
        input_img = imgs[0].to(device).unsqueeze(0).requires_grad_(True)
        
        # Get the Layer Attribution
        attr = engine.get_layer_attribution(input_img)
        
        # Save simple plot
        attr_np = attr.squeeze().cpu().detach().numpy()
        plt.imshow(imgs[0].permute(1,2,0).numpy()) # Original Image
        plt.imshow(attr_np, cmap='jet', alpha=0.5)  # Overlap Heatmap
        plt.title(f"Layer XAI: {img_id}")
        plt.savefig(os.path.join(out_dir, f"{img_id}_layer.png"))
        plt.close()

if __name__ == "__main__":
    main()