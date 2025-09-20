# src/xai/captum/run_captum_det_roi.py
import os, json, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from captum.attr import IntegratedGradients

from src.models.fasterrcnn_cpu import build_fasterrcnn_cpu

OUT_DIR      = "outputs/xai_det"                 # where *_xai_header.json live
CKPT_PATH    = "outputs/fasterrcnn_squares_cpu.pt"
STEPS_IG     = 64
ALPHA        = 0.5

def tfm_eval(pil):
    import torchvision.transforms as T
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406],
                    std=[0.229,0.224,0.225])
    ])(pil)

def denorm(x):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    return (x*std + mean).clamp(0,1)

def overlay(img_chw, hm_hw, alpha=0.5):
    img = img_chw.detach().permute(1,2,0).cpu().numpy()
    cm  = plt.cm.jet(hm_hw.detach().cpu().numpy())[..., :3]
    return np.clip((1-alpha)*img + alpha*cm, 0, 1)

def headers():
    return sorted(glob.glob(os.path.join(OUT_DIR, "*_xai_header.json")))

def make_forward_fn(model, W, H, box_xyxy):
    """
    Return f(x: [B,3,H,W] normalized) -> [B] logits for ROI class=1.
    Handles Captum's batched calls during Integrated Gradients.
    """
    device = next(model.parameters()).device
    x1, y1, x2, y2 = box_xyxy

    def forward_fn(x_in):
        # x_in: [B,3,H,W]
        assert x_in.dim() == 4 and x_in.size(1) == 3, f"Expected [B,3,H,W], got {x_in.shape}"
        B, C, H_in, W_in = x_in.shape

        # Convert batch to list of 3D tensors for transform
        imgs_list = [x_in[b].to(device) for b in range(B)]
        images_list, _ = model.transform(imgs_list, None)     # ImageList (batched)
        x = images_list.tensors                               # [B,3,Ht,Wt]
        image_sizes = images_list.image_sizes                 # list of (Ht, Wt), len B

        # Backbone features (OrderedDict of levels); forward is batched
        features_od = model.backbone(x)

        # Map the same ROI box to transformed coords for each image in batch
        boxes_tr = []
        for b in range(B):
            Ht, Wt = image_sizes[b]
            sx, sy = Wt / float(W), Ht / float(H)
            boxes_tr.append(torch.tensor([[x1*sx, y1*sy, x2*sx, y2*sy]],
                                         dtype=torch.float32, device=device))

        # MultiScaleRoIAlign expects a list length B
        pooled = model.roi_heads.box_roi_pool(features_od, boxes_tr, image_sizes)  # [B,C,7,7]
        rep = model.roi_heads.box_head(pooled)                                     # [B,1024]
        class_logits, _ = model.roi_heads.box_predictor(rep)                       # [B,num_classes]
        return class_logits[:, 1]  # logit for class=1 (square), shape [B]

    return forward_fn

def main():
    device = torch.device("cpu")
    model = build_fasterrcnn_cpu(num_classes=2).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    hdrs = headers()
    if not hdrs:
        print(f"No headers found in {OUT_DIR}. Run the sandbox first.")
        return

    for hpath in hdrs:
        with open(hpath, "r") as f:
            h = json.load(f)
        img_path = h["image"]
        roi = h.get("xai_targets", {}).get("roi", None)
        if roi is None or "box" not in roi:
            print(f"[{Path(img_path).stem}] no ROI in header; skip")
            continue

        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        x = tfm_eval(pil).unsqueeze(0).to(device)   # [1,3,H,W] normalized (H,W are original)
        x.requires_grad_(True)

        fwd = make_forward_fn(model, W, H, roi["box"])
        ig = IntegratedGradients(fwd)

        baseline = torch.zeros_like(x)              # [1,3,H,W] black baseline
        # Captum will internally expand to [STEPS,3,H,W]; fwd handles batching
        attr = ig.attribute(x, baselines=baseline, n_steps=STEPS_IG)  # [1,3,H,W]
        attr = attr.abs().squeeze(0).sum(0)        # [H,W]
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-6)

        over = overlay(denorm(x[0].detach()), attr, alpha=ALPHA)

        stem = Path(img_path).stem
        out_png = os.path.join(OUT_DIR, f"{stem}_captum_roi_ig.png")

        # draw ROI box
        fig = plt.figure(figsize=(4.6,4.6))
        plt.imshow(over); plt.axis("off")
        x1,y1,x2,y2 = roi["box"]
        plt.gca().add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1,
                                          fill=False, linewidth=1.5, edgecolor="white"))
        plt.title("Captum Integrated Gradients toward ROI class=1", fontsize=9)
        plt.tight_layout(); plt.savefig(out_png, dpi=160, bbox_inches="tight"); plt.close(fig)
        print("Saved", Path(out_png).name)

    print("Done. Captum (detector ROI) results in", OUT_DIR)

if __name__ == "__main__":
    main()
