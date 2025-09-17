import numpy as np
import torch
from torchvision.ops import box_iou

@torch.no_grad()
def map_50(model, loader, device):
    model.eval()
    tp, fp, scores, npos = [], [], [], 0
    for imgs, targets in loader:
        imgs = [img.to(device) for img in imgs]
        gts  = [t["boxes"].to(device) for t in targets]
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
                fp.extend([1]*len(boxes)); tp.extend([0]*len(boxes)); continue
            ious = box_iou(boxes, gt)
            matched = set()
            for i in range(boxes.size(0)):
                j = torch.argmax(ious[i]).item()
                if ious[i, j] >= 0.5 and j not in matched:
                    tp.append(1); fp.append(0); matched.add(j)
                else:
                    tp.append(0); fp.append(1)
    if len(scores)==0: return 0.0
    idx = np.argsort(-np.array(scores))
    tp = np.array(tp)[idx]; fp = np.array(fp)[idx]
    tp_cum = np.cumsum(tp); fp_cum = np.cumsum(fp)
    recall = tp_cum / max(1, npos)
    precision = tp_cum / np.maximum(1, tp_cum + fp_cum)
    ap = 0.0
    for r in np.linspace(0,1,11):
        p = np.max(precision[recall>=r]) if np.any(recall>=r) else 0
        ap += p / 11.0
    return float(ap)
