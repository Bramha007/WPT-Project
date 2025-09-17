from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from .voc_parser import parse_voc


class SquaresClassificationDataset(Dataset):
    def __init__(
        self, pairs, canvas=128, train=True, use_padding_canvas=True, margin_px=16
    ):
        self.samples = []
        self.canvas = canvas
        self.train = train
        self.use_padding_canvas = use_padding_canvas
        self.margin_px = margin_px

        # NOTE: Do not include random scaling; we want to preserve scale cues.
        tfms = [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if train:
            tfms = [T.RandomHorizontalFlip()] + tfms
        self.transform = T.Compose(tfms)

        for img_path, xml_path in pairs:
            rec = parse_voc(xml_path)
            img = Image.open(img_path).convert("RGB")
            W, H = img.size
            for x1, y1, x2, y2 in rec["boxes"]:
                # Ensure square side (use width=height in your data)
                side = max(x2 - x1, y2 - y1)
                label = self.size_to_class(side)

                if self.use_padding_canvas:
                    # ---- Approach A: paste onto fixed canvas (no scaling) ----
                    crop = img.crop((x1, y1, x2, y2))
                    canvas = Image.new(
                        "RGB", (self.canvas, self.canvas), (255, 255, 255)
                    )
                    # center paste
                    ox = (self.canvas - crop.size[0]) // 2
                    oy = (self.canvas - crop.size[1]) // 2
                    canvas.paste(crop, (ox, oy))
                    sample_img = canvas
                else:
                    # ---- Approach B: fixed pixel margin then resize ----
                    x1m = max(0, x1 - self.margin_px)
                    y1m = max(0, y1 - self.margin_px)
                    x2m = min(W, x2 + self.margin_px)
                    y2m = min(H, y2 + self.margin_px)
                    crop = img.crop((x1m, y1m, x2m, y2m))
                    sample_img = crop.resize((self.canvas, self.canvas), Image.BILINEAR)

                self.samples.append((sample_img, label))

    def size_to_class(self, side):
        # Use exact bins that match your generator (8,16,32,64,128).
        # Map by nearest instead of crude thresholds.
        bins = [8, 16, 32, 64, 128]
        # pick nearest bin
        idx = min(range(len(bins)), key=lambda i: abs(side - bins[i]))
        return idx

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return self.transform(img), label