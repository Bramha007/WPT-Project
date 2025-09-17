# src/train/eval_det.py
import torch
from torch.utils.data import DataLoader
from src.dataio.det_dataset import QuadrilateralDetectionDataset, collate_fn
from src.dataio.det_transforms import Compose, ToTensor
from src.models.fasterrcnn import build_fasterrcnn
from .train_det import evaluate_map_50

def eval_on_test(test_pairs, ckpt="fasterrcnn_quadrilateral.pt", device="cuda"):
    ds = QuadrilateralDetectionDataset(test_pairs, transforms=Compose([ToTensor()]))
    dl = DataLoader(ds, batch_size=2, shuffle=False, num_workers=2, collate_fn=collate_fn)
    model = build_fasterrcnn(num_classes=2).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    ap50 = evaluate_map_50(model, dl, device)
    print(f"Test (rectangles) mAP@0.5: {ap50:.3f}")
