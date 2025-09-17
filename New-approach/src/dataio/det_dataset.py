import torch
from torch.utils.data import Dataset
from PIL import Image
from .voc_parser import parse_voc


class SquaresDetectionDataset(Dataset):
    def __init__(self, pairs, transforms=None):
        self.pairs = pairs
        self.transforms = transforms

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, xml_path = self.pairs[idx]
        img = Image.open(img_path)
        rec = parse_voc(xml_path)

        boxes = torch.tensor(rec["boxes"], dtype=torch.float32)
        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)  # single class = 1
        image_id = torch.tensor([idx])
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))
