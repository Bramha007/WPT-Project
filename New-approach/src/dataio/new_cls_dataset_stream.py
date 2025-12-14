from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import List, Tuple
from .voc_parser import parse_voc
from src.setup.new_config_cls import AREA_BINS
import torch

class GeometricShapeClassificationDatasetStream(Dataset):
    """
    Dataset for Multi-Task Classification (Area) + Regression (W/H).
    Returns a sample: (image_tensor, y_area_id, y_wh_target_tensor).
    """
    def __init__(self, pairs: List[Tuple[str,str]], canvas=224, train=True,
                 use_padding_canvas=True, margin_px=16):
        self.canvas = canvas
        self.train = train
        self.use_padding_canvas = use_padding_canvas
        self.margin_px = margin_px

        self.index = []  
        
        for img_path, xml_path in pairs:
            rec = parse_voc(xml_path)
            W_img, H_img = rec["width"], rec["height"]
            
            if "labels_wxh_str" not in rec or len(rec["labels_wxh_str"]) == 0:
                continue

            for (x1, y1, x2, y2), wxh_str in zip(rec["boxes"], rec["labels_wxh_str"]):
                
                box_w = x2 - x1
                box_h = y2 - y1
                area = box_w * box_h 
                
                # LABEL: Area Class ID (0-4)
                y_area_id = self._area_to_class(area)
                
                # TARGET: W and H in pixels
                # Using the raw box dimensions (W, H) as the regression target
                self.index.append((img_path, x1, y1, x2, y2, y_area_id, box_w, box_h, W_img, H_img))


        # AUGMENTATION PIPELINE (Synthetic Rectangles)
        tfms = [T.ToTensor(), T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])]
        
        if train:
            geom_tfms = [
                T.RandomAffine(
                    degrees=45,         
                    shear=[-20, 20],    
                    scale=(0.8, 1.2)    
                ),
                T.RandomHorizontalFlip() 
            ]
            tfms = geom_tfms + tfms
            
        self.transform = T.Compose(tfms)
        
        print(f"DEBUG: Dataset initialization finished. Total items in dataset: {len(self.index)}")


    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # Retrieving the image, Area label, and W/H targets
        img_path, x1, y1, x2, y2, y_area_id, box_w, box_h, W_img, H_img = self.index[idx]
        img = Image.open(img_path).convert("RGB")

        # --- Object Cropping Logic (unchanged) ---
        if self.use_padding_canvas:
            crop = img.crop((x1, y1, x2, y2))
            canvas = Image.new("RGB", (self.canvas, self.canvas), (255,255,255))
            ox = (self.canvas - crop.size[0]) // 2
            oy = (self.canvas - crop.size[1]) // 2
            canvas.paste(crop, (ox, oy))
            sample_img = canvas
        else:
            # ... (cropping with margin) ...
            pass
            
        # Create the W/H target tensor [W, H]
        y_wh_target = torch.tensor([box_w, box_h], dtype=torch.float32)
            
        # Return the transformed image, the Area Classification ID, and the W/H Regression Target
        return self.transform(sample_img), y_area_id, y_wh_target

    @staticmethod
    def _area_to_class(area: int) -> int:
        bins = AREA_BINS 
        return min(range(len(bins)), key=lambda i: abs(area - bins[i]))