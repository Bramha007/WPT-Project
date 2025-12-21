import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from .voc_parser import parse_voc
from src.setup import new_config_cls as config

class GeometricShapeClassificationDatasetStream(Dataset):
    def __init__(self, pairs, train=True):
        self.pairs = pairs
        self.train = train
        self.index = []  

        for img_path, xml_path in pairs:
            rec = parse_voc(xml_path)
            for (x1, y1, x2, y2) in rec["boxes"]:
                area = (x2 - x1) * (y2 - y1)
                label_id = self._area_to_class(area)
                self.index.append((img_path, x1, y1, x2, y2, label_id))

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        img_path, x1, y1, x2, y2, label_id = self.index[idx]
        img = Image.open(img_path).convert("RGB")
        crop = img.crop((x1, y1, x2, y2))

        # --- DYNAMIC RECTANGLE SYNTHESIS ---
        if self.train:
            w, h = crop.size
            # Randomly distort aspect ratio between 0.5 (tall) and 2.0 (wide)
            ar = np.random.uniform(0.5, 2.0)
            new_w, new_h = int(w * ar), int(h / ar)
            # Clip to ensure the synthesized shape fits the 224x224 canvas
            new_w, new_h = min(max(new_w, 8), 200), min(max(new_h, 8), 200)
            crop = crop.resize((new_w, new_h), Image.BILINEAR)

        canvas = Image.new("RGB", (config.CANVAS_SIZE, config.CANVAS_SIZE), (255, 255, 255))
        canvas.paste(crop, ((config.CANVAS_SIZE - crop.size[0]) // 2, (config.CANVAS_SIZE - crop.size[1]) // 2))
        return self.transform(canvas), label_id

    @staticmethod
    def _area_to_class(area):
        for i, b in enumerate(config.AREA_BOUNDARIES):
            if area < b: return i
        return len(config.AREA_BOUNDARIES)