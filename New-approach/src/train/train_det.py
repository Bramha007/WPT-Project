# src/train/train_det.py
import torch, time
from torch.utils.data import DataLoader, random_split
from torchvision.ops import box_iou
from src.dataio.det_dataset import QuadrilateralDetectionDataset, collate_fn
from src.dataio.det_transforms import Compose, ToTensor, RandomHorizontalFlip
from src.models.fasterrcnn import build_fasterrcnn

def make_loaders(pairs, val_ratio=0.1, bs=2):
    n = len(pairs); n_val = max(1, int(n*val_ratio))
    train_pairs, val_pairs = pairs[:-n_val], pairs[-n_val:]

    train_ds = QuadrilateralDetectionDataset(
        train_pairs, transforms=Compose([ToTensor(), RandomHorizontalFlip(0.5)])
    )
    val_ds   = QuadrilateralDetectionDataset(
        val_pairs, transforms=Compose([ToTensor()])
    )
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                          num_workers=2, collate_fn=collate_fn)
    val_dl   = DataLoader(val_ds, batch_size=bs, shuffle=False,
                          num_workers=2, collate_fn=collate_fn)
    return train_dl, val_dl

@torch.no_grad()
def evaluate_map_50(model, loader, device):
    model.eval()
    tp, fp, scores, npos = [], [], [], 0
    iou_thr = 0.5
    for imgs, targets in loader:
        imgs = [img.to(device) for img in imgs]
        gts = [t["boxes"].to(device) for t in targets]
        npos += sum(len(x) for x in gts)
        preds = model(imgs)
        for pred, gt in zip(preds, gts):
            if len(pred["boxes"]) == 0:
                continue
            boxes = pred["boxes"]; conf = pred["scores"]
            order = torch.argsort(conf, descending=True)
            boxes, conf = boxes[order], conf[order]
            scores.extend(conf.cpu().tolist())
            if len(gt)==0:
                fp.extend([1]*len(boxes)); tp.extend([0]*len(boxes))
                continue
            ious = box_iou(boxes, gt)  # [Nd, Ng]
            matched_gt = set()
            for i in range(boxes.size(0)):
                j = torch.argmax(ious[i]).item()
                if ious[i, j] >= iou_thr and j not in matched_gt:
                    tp.append(1); fp.append(0); matched_gt.add(j)
                else:
                    tp.append(0); fp.append(1)
    if len(scores)==0: return 0.0
    # sort by score
    import numpy as np
    idx = np.argsort(-np.array(scores))
    tp = np.array(tp)[idx]; fp = np.array(fp)[idx]
    tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
    recall = tp_cum / max(1, npos)
    precision = tp_cum / np.maximum(1, tp_cum + fp_cum)
    # AP via 11-pt interpolation
    ap = 0.0
    for r in np.linspace(0,1,11):
        p = np.max(precision[recall>=r]) if np.any(recall>=r) else 0
        ap += p / 11.0
    return float(ap)

def train(pairs, epochs=20, bs=2, lr=0.005, device="cpu"):
    train_dl, val_dl = make_loaders(pairs, val_ratio=0.1, bs=bs)
    model = build_fasterrcnn(num_classes=2).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=16, gamma=0.1)

    best_ap, best_path = 0.0, "fasterrcnn_quadrilateral.pt"
    for epoch in range(epochs):
        model.train()
        loss_sum, t0 = 0.0, time.time()
        for imgs, targets in train_dl:
            imgs = [img.to(device) for img in imgs]
            targets = [{k:v.to(device) for k,v in t.items()} for t in targets]
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item()
        sched.step()
        ap50 = evaluate_map_50(model, val_dl, device)
        print(f"Epoch {epoch+1:02d}/{epochs} | loss {loss_sum/len(train_dl):.4f} | mAP@0.5 {ap50:.3f} | {time.time()-t0:.1f}s")
        if ap50 > best_ap:
            best_ap = ap50
            torch.save(model.state_dict(), best_path)
    print(f"Best val mAP@0.5: {best_ap:.3f}; saved to {best_path}")
