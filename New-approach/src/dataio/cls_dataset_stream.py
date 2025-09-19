from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from typing import List, Tuple
from .voc_parser import parse_voc

class SquaresClassificationDatasetStream(Dataset):
    """
    Streaming version: does NOT prebuild crops in memory.
    Keeps a flat list of (img_path, (x1,y1,x2,y2), label).
    Crops on-the-fly in __getitem__.
    """
    def __init__(self, pairs: List[Tuple[str,str]], canvas=224, train=True,
                 use_padding_canvas=True, margin_px=16):
        self.canvas = canvas
        self.train = train
        self.use_padding_canvas = use_padding_canvas
        self.margin_px = margin_px

        # Build a flat index of all boxes for the given (img, xml) pairs.
        self.index = []  # list of tuples: (img_path, x1,y1,x2,y2, label, W,H)
        for img_path, xml_path in pairs:
            rec = parse_voc(xml_path)
            W, H = rec["width"], rec["height"]
            for (x1, y1, x2, y2) in rec["boxes"]:
                side = max(x2-x1, y2-y1)
                label = self._size_to_class(side)
                self.index.append((img_path, x1, y1, x2, y2, label, W, H))

        tfms = [T.ToTensor(),
                T.Normalize(mean=[0.485,0.456,0.406],
                            std=[0.229,0.224,0.225])]
        if train:
            tfms = [T.RandomHorizontalFlip()] + tfms
        self.transform = T.Compose(tfms)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        img_path, x1, y1, x2, y2, label, W, H = self.index[idx]
        img = Image.open(img_path).convert("RGB")

        if self.use_padding_canvas:
            crop = img.crop((x1, y1, x2, y2))
            canvas = Image.new("RGB", (self.canvas, self.canvas), (255,255,255))
            # center paste (optionally jitter a few pixels during training)
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

        return self.transform(sample_img), label

    @staticmethod
    def _size_to_class(side):
        bins = [8, 16, 32, 64, 128]
        return min(range(len(bins)), key=lambda i: abs(side - bins[i]))
