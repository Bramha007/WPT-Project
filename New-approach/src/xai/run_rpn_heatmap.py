# src/xai/run_rpn_heatmap.py
import os, json, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from src.models.fasterrcnn_cpu import build_fasterrcnn_cpu

# --------- Paths / config (edit if needed) ----------
OUT_DIR = "outputs/xai_det"  # where A1.1 artifacts live
CKPT_PATH = "outputs/fasterrcnn_squares_cpu.pt"
ALPHA = 0.45  # heatmap overlay strength
# ----------------------------------------------------


# --- tiny helpers ---
def to_tensor_norm(pil):
    import torchvision.transforms as T

    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(pil)


def denorm(img):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img * std + mean).clamp(0, 1)


def overlay_img(img_chw, hm_hw, alpha=0.45):
    """img_chw:[3,H,W] float(0..1), hm_hw:[H,W] float(0..1)"""
    img = img_chw.detach().permute(1, 2, 0).cpu().numpy()
    cm = plt.cm.jet(hm_hw.detach().cpu().numpy())[..., :3]
    out = (1 - alpha) * img + alpha * cm
    return np.clip(out, 0, 1)


def read_header_jsons(out_dir):
    return sorted(glob.glob(os.path.join(out_dir, "*_xai_header.json")))


def load_image_tensor_from_header(header):
    pil = Image.open(header["image"]).convert("RGB")
    tens = to_tensor_norm(pil)
    W, H = pil.size
    return tens, (W, H)


def box_center(box):
    x1, y1, x2, y2 = box
    return (0.5 * (x1 + x2), 0.5 * (y1 + y2))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cpu")

    # 1) Build & load detector
    model = build_fasterrcnn_cpu(num_classes=2).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    # 2) Loop over headers from A1.1
    headers = read_header_jsons(OUT_DIR)
    if not headers:
        print(f"No *_xai_header.json found in {OUT_DIR}. Run the sandbox first.")
        return

    for hpath in headers:
        with open(hpath, "r") as f:
            header = json.load(f)

        img_path = header["image"]
        stem = Path(img_path).stem
        save_png = os.path.join(OUT_DIR, f"{stem}_rpn_heatmap.png")

        # Load and prep image
        img_t, (W, H) = load_image_tensor_from_header(header)

        # Use the same internal transform as the detector for alignment
        images_list, _ = model.transform([img_t.to(device)], None)  # ImageList
        x = images_list.tensors  # [1,3,Ht,Wt]
        Ht, Wt = images_list.image_sizes[0]

        # ---- Forward through backbone+FPN; capture features as a LIST ----
        with torch.enable_grad():
            features_od = model.backbone(
                x
            )  # OrderedDict: level_name -> Tensor[1,C,Hl,Wl]
            level_names = list(features_od.keys())
            feat_list = list(
                features_od.values()
            )  # List[Tensor], aligned with level_names

            # keep grads on tensors that the RPN head will actually consume
            for t in feat_list:
                t.retain_grad()

            # RPN head expects a *list* of tensors
            objectness_list, bbox_deltas_list = model.rpn.head(
                feat_list
            )  # lists aligned to feat_list

            # Use ROI target box center from header to pick the cell/anchor to explain
            roi_tgt = header.get("xai_targets", {}).get("roi", None)
            if roi_tgt is None or "box" not in roi_tgt:
                print(f"[{stem}] No ROI target in header; skipping.")
                continue
            roi_box = roi_tgt["box"]
            xc, yc = box_center(roi_box)

            # Map (xc,yc) to each level; choose (level,anchor,cell) with highest objectness at that cell
            best = None  # (score, level_idx, anchor_idx, i, j)
            for l_idx, feat in enumerate(feat_list):
                _, C, Hl, Wl = feat.shape
                # strides based on transformed (Ht,Wt)
                stride_y = float(Ht) / Hl
                stride_x = float(Wt) / Wl
                j = int(np.clip(np.floor(xc / stride_x), 0, Wl - 1))
                i = int(np.clip(np.floor(yc / stride_y), 0, Hl - 1))

                logits = objectness_list[l_idx][0]  # [A, Hl, Wl]
                cell_logits = logits[:, i, j]  # [A]
                a_idx = int(torch.argmax(cell_logits).item())
                s = float(cell_logits[a_idx].item())
                if (best is None) or (s > best[0]):
                    best = (s, l_idx, a_idx, i, j)

            if best is None:
                print(f"[{stem}] Could not select an RPN target; skipping.")
                continue

            s, l_idx, a_idx, i, j = best

            # Backprop a single scalar (pre-sigmoid objectness logit)
            scalar = objectness_list[l_idx][0, a_idx, i, j]
            model.zero_grad()
            for t in feat_list:
                if t.grad is not None:
                    t.grad.zero_()
            scalar.backward(retain_graph=False)

            # Grad-CAM on that level (IMPORTANT: get grads from the retained tensor, not the slice)
            Feat4cam = feat_list[l_idx][0]  # [C, Hl, Wl] values
            Grad4cam = feat_list[l_idx].grad[0]  # [C, Hl, Wl] grads live here

            if Grad4cam is None:
                print(
                    f"[{stem}] No gradients captured on level {level_names[l_idx]} — skipping."
                )
                continue

            weights = Grad4cam.mean(dim=(1, 2), keepdim=True)  # [C,1,1]
            cam = (weights * Feat4cam).sum(dim=0)  # [Hl, Wl]
            cam = F.relu(cam)

            # Upsample CAM to transformed size (Ht,Wt), then to original (H,W)
            cam = cam.unsqueeze(0).unsqueeze(0)  # [1,1,Hl,Wl]
            cam_up = F.interpolate(
                cam, size=(Ht, Wt), mode="bilinear", align_corners=False
            )[
                0, 0
            ]  # [Ht,Wt]
            cam_img = F.interpolate(
                cam_up.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

            # Normalize [0,1]
            cam_img = cam_img - cam_img.min()
            cam_img = cam_img / (cam_img.max() + 1e-6)

        # ----- Visualize overlay -----
        img_vis = denorm(img_t).cpu()
        over = overlay_img(img_vis, cam_img, alpha=ALPHA)

        # Draw the ROI target box for context (white rectangle)
        fig = plt.figure(figsize=(4.6, 4.6))
        plt.imshow(over)
        plt.axis("off")
        roi = header.get("xai_targets", {}).get("roi", None)
        if roi is not None and "box" in roi:
            x1, y1, x2, y2 = roi["box"]
            plt.gca().add_patch(
                plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    fill=False,
                    linewidth=1.5,
                    edgecolor="white",
                )
            )
        plt.title(f"RPN Grad-CAM @ {level_names[l_idx]}  obj_logit={s:.2f}", fontsize=9)
        plt.tight_layout()
        plt.savefig(save_png, dpi=160, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved RPN heatmap → {Path(save_png).name}")

    print("Done: RPN heatmaps in", OUT_DIR)


if __name__ == "__main__":
    main()
