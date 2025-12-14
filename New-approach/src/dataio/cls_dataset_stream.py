# File: src/dataio/cls_dataset_stream.py (FINAL COMPLETE VERSION)

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import List, Tuple
from .voc_parser import parse_voc
# Import the full WxH class map from the setup configuration
from src.setup.config_cls import SIZE_CLASS_MAP 

class GeometricShapeClassificationDatasetStream(Dataset):
    """
    Dataset for multi-class WxH size classification.
    Crops the object on-the-fly, applies robust augmentation (including synthetic stretching),
    and assigns a unique WxH class label (0-24).
    """
    def __init__(self, pairs: List[Tuple[str,str]], canvas=224, train=True,
                 use_padding_canvas=True, margin_px=16):
        self.canvas = canvas
        self.train = train
        self.use_padding_canvas = use_padding_canvas
        self.margin_px = margin_px

        # Build a flat index of all boxes for the given (img, xml) pairs.
        # list of tuples: (img_path, x1,y1,x2,y2, label_ID, W,H)
        self.index = []  
        
        # --- DEBUG: Initialize counters for integrity check ---
        total_objects_added = 0
        
        for img_path, xml_path in pairs:
            # We rely on parse_voc returning 'labels_wxh_str'
            rec = parse_voc(xml_path)
            W, H = rec["width"], rec["height"]
            
            # CRITICAL CHECK: Ensure the WxH string labels are present
            if "labels_wxh_str" not in rec or len(rec["labels_wxh_str"]) == 0:
                continue

            # Iterate over boxes and their corresponding WxH string labels
            for (x1, y1, x2, y2), wxh_str in zip(rec["boxes"], rec["labels_wxh_str"]):
                
                # --- NEW: Convert WxH string to integer class ID (0-24) ---
                try:
                    label_id = SIZE_CLASS_MAP[wxh_str]
                except KeyError:
                    # Skip samples with WxH combinations not defined in our map
                    continue 
                
                self.index.append((img_path, x1, y1, x2, y2, label_id, W, H))
                total_objects_added += 1

        # --- AUGMENTATION PIPELINE ---
        
        # 1. Base transforms (ToTensor and Normalization)
        tfms = [
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406],
                        std=[0.229,0.224,0.225])
        ]
        
        if train:
            # 2. Geometric Augmentations (Synthetic Rectangles and Orientation Invariance)
            
            # T.RandomAffine: CRITICAL AUGMENTATION for generalization.
            # - degrees=45: Random Rotation (Breaks orientation bias)
            # - shear=[-20, 20]: Random Shear (Synthesizes rectangular stretching/tilting)
            # - scale=(0.8, 1.2): Slight random zoom
            geom_tfms = [
                T.RandomAffine(
                    degrees=45,         
                    shear=[-20, 20],    
                    scale=(0.8, 1.2)    
                ),
                T.RandomHorizontalFlip() # Standard Flip
            ]
            
            # Insert geometric transforms BEFORE ToTensor/Normalization
            tfms = geom_tfms + tfms
            
        self.transform = T.Compose(tfms)
        
        # DEBUG: Print final count to confirm the 25-class data was loaded
        print(f"DEBUG: Dataset initialization finished. Total items in dataset: {len(self.index)}")


    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # Retrieve data, including the correct label_ID
        img_path, x1, y1, x2, y2, label_id, W, H = self.index[idx]
        img = Image.open(img_path).convert("RGB")

        # --- Object Cropping Logic ---
        if self.use_padding_canvas:
            crop = img.crop((x1, y1, x2, y2))
            canvas = Image.new("RGB", (self.canvas, self.canvas), (255,255,255))
            ox = (self.canvas - crop.size[0]) // 2
            oy = (self.canvas - crop.size[1]) // 2
            canvas.paste(crop, (ox, oy))
            sample_img = canvas
        else:
            x1m = max(0, x1 - self.margin_px)
            y1m = max(0, y1 - self.margin_px)
            x2m = min(W, x2 + self.margin_px)
            y2m = min(H, y2 + self.margin_px)
            crop = img.crop((x1m, y1m, x2m, y2m))
            sample_img = crop.resize((self.canvas, self.canvas), Image.BILINEAR)

        # Apply transformation pipeline and return image tensor and integer label_ID
        return self.transform(sample_img), label_id

    # The old _size_to_class static method is REMOVED.