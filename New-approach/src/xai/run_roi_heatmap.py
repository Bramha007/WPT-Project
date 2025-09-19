# src/xai/run_roi_heatmap.py
import os, json, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from src.models.fasterrcnn_cpu import build_fasterrcnn_cpu

OUT_DIR   = "outputs/xai_det"                 # where *_xai_header.json live
CKPT_PATH = "outputs/fasterrcnn_squares_cpu.pt"
ALPHA     = 0.45                              # overlay strength
POOLED    = (7, 7)                            # ROIAlign pooled size (default)

# ---------------- helpers ----------------
def to_tensor_norm(pil):
    import torchvision.transforms as T
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])(pil)

def denorm(img):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    return (img*std + mean).clamp(0,1)

def overlay_img(img_chw, hm_hw, alpha=0.45):
    img = img_chw.detach().permute(1,2,0).cpu().numpy()
    cm  = plt.cm.jet(hm_hw.detach().cpu().numpy())[..., :3]
    out = (1-alpha)*img + alpha*cm
    return np.clip(out, 0, 1)

def read_header_jsons(out_dir):
    return sorted(glob.glob(os.path.join(out_dir, "*_xai_header.json")))

def load_pil_and_tensor(img_path):
    pil = Image.open(img_path).convert("RGB")
    tens = to_tensor_norm(pil)
    return pil, tens

# --------------- main --------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cpu")

    model = build_fasterrcnn_cpu(num_classes=2).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    headers = read_header_jsons(OUT_DIR)
    if not headers:
        print(f"No *_xai_header.json found in {OUT_DIR}. Run the sandbox first.")
        return

    for hpath in headers:
        with open(hpath, "r") as f:
            header = json.load(f)

        img_path = header["image"]
        roi_tgt  = header.get("xai_targets", {}).get("roi", None)
        if roi_tgt is None or "box" not in roi_tgt:
            print(f"[{Path(img_path).stem}] No ROI target; skipping.")
            continue

        stem = Path(img_path).stem
        save_png = os.path.join(OUT_DIR, f"{stem}_roi_heatmap.png")

        pil, img_t = load_pil_and_tensor(img_path)
        W, H = pil.size

        # Transform image exactly like the detector
        images_list, _ = model.transform([img_t.to(device)], None)
        x = images_list.tensors
        Ht, Wt = images_list.image_sizes[0]

        # Backbone+FPN
        with torch.enable_grad():
            features_od = model.backbone(x)         # OrderedDict: name -> Tensor[1,C,Hl,Wl]
            image_shapes = images_list.image_sizes

            # --- prepare the single ROI box in transformed coordinates ---
            x1,y1,x2,y2 = roi_tgt["box"]
            # guard for any tiny numeric noise
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(W), x2), min(float(H), y2)

            scale_x = float(Wt) / float(W)
            scale_y = float(Ht) / float(H)
            box_tr  = torch.tensor([[x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y]],
                                   dtype=torch.float32, device=device)  # [1,4]
            boxes_per_image = [box_tr]   # List[Tensor] size 1

            # --- ROIAlign pooled feature (PASS THE ORDEREDDICT!) ---
            pooled = model.roi_heads.box_roi_pool(features_od, boxes_per_image, image_shapes)  # [1,C,7,7]
            pooled.retain_grad()

            # Forward through ROI box head + predictor
            rep = model.roi_heads.box_head(pooled)          # [1,1024] (TwoMLPHead)
            class_logits, bbox_deltas = model.roi_heads.box_predictor(rep)

            # We have 2 classes: background=0, square=1
            scalar = class_logits[0, 1]                     # foreground logit
            model.zero_grad()
            if pooled.grad is not None:
                pooled.grad.zero_()
            scalar.backward(retain_graph=False)

            # Grad-CAM on pooled (7x7) map
            # pooled.grad shape: [1,C,7,7]; pooled[0] shape: [C,7,7]
            Grad = pooled.grad[0]
            Feat = pooled[0]
            weights = Grad.mean(dim=(1,2), keepdim=True)    # [C,1,1]
            cam = (weights * Feat).sum(dim=0)               # [7,7]
            cam = F.relu(cam)

            # Upsample to the ROI box size (in original image pixels)
            box_w = max(1, int(round(x2 - x1)))
            box_h = max(1, int(round(y2 - y1)))
            cam_up = F.interpolate(cam.unsqueeze(0).unsqueeze(0),
                                   size=(box_h, box_w), mode="bilinear",
                                   align_corners=False)[0,0]

            # Place on full canvas
            hm = torch.zeros((H, W), dtype=torch.float32)
            x1i, y1i = int(round(x1)), int(round(y1))
            x2i, y2i = min(W, x1i+box_w), min(H, y1i+box_h)
            # clip cam_up if needed (edge cases)
            cut_h = y2i - y1i
            cut_w = x2i - x1i
            hm[y1i:y2i, x1i:x2i] = cam_up[:cut_h, :cut_w]

            # Normalize [0,1]
            hm = hm - hm.min()
            hm = hm / (hm.max() + 1e-6)

        # Overlay
        img_vis = denorm(img_t).cpu()
        over = overlay_img(img_vis, hm, alpha=ALPHA)

        # Draw the ROI box
        fig = plt.figure(figsize=(4.6, 4.6))
        plt.imshow(over); plt.axis("off")
        plt.gca().add_patch(plt.Rectangle((x1,y1), x2-x1, y2-y1,
                                          fill=False, linewidth=1.5, edgecolor="white"))
        plt.title(f"ROI Grad-CAM (cls=1)  logit={float(scalar.item()):.2f}", fontsize=9)
        plt.tight_layout()
        plt.savefig(save_png, dpi=160, bbox_inches="tight"); plt.close(fig)
        print(f"Saved ROI heatmap → {Path(save_png).name}")

    print("Done: ROI heatmaps in", OUT_DIR)


if __name__ == "__main__":
    main()
