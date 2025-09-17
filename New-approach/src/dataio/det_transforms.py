import random
from torchvision.transforms import functional as F

class ToTensor:
    def __call__(self, img, target):
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = F.to_tensor(img)  # float32 [0,1]
        return img, target

class RandomHorizontalFlip:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, img, target):
        if random.random() < self.p:
            w, _ = F.get_image_size(img)
            boxes = target["boxes"].clone()
            boxes[:, [0,2]] = w - boxes[:, [2,0]]
            target["boxes"] = boxes
            img = F.hflip(img)
        return img, target

class Compose:
    def __init__(self, ts): self.ts = ts
    def __call__(self, img, target):
        for t in self.ts: img, target = t(img, target)
        return img, target
