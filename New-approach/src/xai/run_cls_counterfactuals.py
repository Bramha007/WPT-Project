# src/xai/run_cls_counterfactuals.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.cls_dataset_stream import SquaresClassificationDatasetStream
from src.dataio.split_utils import subsample_pairs
from src.models.resnet_cls import build_resnet_classifier
from src.xai.gradcam_utils import GradCAM

CLASS_NAMES = ["8","16","32","64","128"]

# ========= Config (edit paths & fractions) =========
DATA_ROOT   = r"E:\WPT-Project\Data\sized_squares_filled"   # <-- your dataset root
IMG_DIR_TEST = fr"{DATA_ROOT}\test"
XML_DIR_ALL  = fr"{DATA_ROOT}\annotations"

CANVAS = 224
CKPT   = "outputs/resnet_cls.pt"
OUTDIR = "outputs/xai_cls"

# Fractioning (apply to TEST in this XAI script)
F_TEST = 0.05   # use 5% of test set; bump as needed
SEED   = 42
# ===================================================

def _denorm(img):
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    return (img * std + mean).clamp(0,1)

def _overlay(img_hw, heatmap_hw, alpha=0.45):
    img = img_hw.permute(1,2,0).cpu().numpy()      # HWC
    hm  = heatmap_hw.squeeze(0).cpu().numpy()      # HxW
    hm  = plt.cm.jet(hm)[..., :3]                  # colorize
    return np.clip((1-alpha)*img + alpha*hm, 0, 1)

def _paste_on_canvas(crop_pil, canvas_size):
    W,H = crop_pil.size
    canvas = Image.new("RGB", (canvas_size, canvas_size), (255,255,255))
    ox = (canvas_size - W)//2
    oy = (canvas_size - H)//2
    canvas.paste(crop_pil, (ox, oy))
    return canvas

def _to_tensor_norm(pil):
    import torchvision.transforms as T
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tfm(pil)

def size_sweep(model, cammer, crop_pil, true_cls, canvas=224, scales=(0.6,0.8,1.0,1.2,1.4), outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    probs = []
    for s in scales:
        new_wh = (max(1,int(crop_pil.size[0]*s)), max(1,int(crop_pil.size[1]*s)))
        resized = crop_pil.resize(new_wh, Image.NEAREST)  # keep edges crisp
        canvas_img = _paste_on_canvas(resized, canvas)
        x = _to_tensor_norm(canvas_img).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            p = logits.softmax(1)[0]
            pred = int(p.argmax().item())
            conf = float(p[pred].item())
        cam, _, _ = cammer(x, target_class=pred)

        title = f"scale {s:.1f}  T:{CLASS_NAMES[true_cls]}  P:{CLASS_NAMES[pred]} ({conf:.2f})"
        plt.figure(figsize=(3,3))
        plt.imshow(_overlay(_denorm(x[0]), cam[0])); plt.axis("off"); plt.title(title, fontsize=9)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"sweep_{s:.1f}.png"), dpi=150, bbox_inches="tight"); plt.close()
        probs.append(conf)

    # prob vs scale plot
    plt.figure(figsize=(4,3))
    plt.plot(scales, probs, marker="o")
    plt.xlabel("scale factor"); plt.ylabel("max prob"); plt.title("Prediction confidence vs. scale")
    plt.ylim(0,1); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "sweep_plot.png"), dpi=150, bbox_inches="tight"); plt.close()

def occlusion_study(model, cammer, canvas_img, true_cls, bar=16, where="top", outdir=OUTDIR):
    os.makedirs(outdir, exist_ok=True)
    fracs = np.linspace(0,1,6)
    probs = []

    for frac in fracs:
        arr = np.array(canvas_img).copy()
        if where == "top":
            h = int(bar*frac); arr[:h, :, :] = 255
        elif where == "bottom":
            h = int(bar*frac); arr[-h:, :, :] = 255
        elif where == "left":
            w = int(bar*frac); arr[:, :w, :] = 255
        else:  # right
            w = int(bar*frac); arr[:, -w:, :] = 255

        occ_img = Image.fromarray(arr)
        x = _to_tensor_norm(occ_img).unsqueeze(0)

        with torch.no_grad():
            logits = model(x)
            p = logits.softmax(1)[0]
            pred = int(p.argmax().item())
            conf = float(p[pred].item())
        cam, _, _ = cammer(x, target_class=pred)

        title = f"occ {where} {frac:.1f}  T:{CLASS_NAMES[true_cls]}  P:{CLASS_NAMES[pred]} ({conf:.2f})"
        plt.figure(figsize=(3,3))
        plt.imshow(_overlay(_denorm(x[0]), cam[0])); plt.axis("off"); plt.title(title, fontsize=9)
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"occl_{where}_{frac:.1f}.png"), dpi=150, bbox_inches="tight"); plt.close()
        probs.append(conf)

    # prob vs occlusion plot
    plt.figure(figsize=(4,3))
    plt.plot(fracs, probs, marker="o")
    plt.xlabel(f"occluded fraction ({where})"); plt.ylabel("max prob"); plt.title("Confidence vs. occlusion")
    plt.ylim(0,1); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, f"occl_plot_{where}.png"), dpi=150, bbox_inches="tight"); plt.close()

def reconstruct_crop_from_canvas(x_tensor):
    """
    Reconstruct the square crop from the canvas tensor by thresholding
    the white background. x_tensor is normalized [3,H,W] float.
    Returns a PIL.Image crop.
    """
    from torchvision.transforms.functional import to_pil_image

    # De-normalize to [0,1] float
    img_f = _denorm(x_tensor)              # float32, [0,1]

    # Grayscale for thresholding (still float)
    gray = img_f.mean(dim=0)               # [H,W], float32

    # Threshold: white background ~1.0; square is dark (~0)
    mask = (gray < 0.78)                   # tweak 0.75–0.85 if needed

    # Fallback if mask is empty (shouldn't happen, but be safe)
    if mask.sum() == 0:
        return to_pil_image(img_f)

    ys, xs = mask.nonzero(as_tuple=True)
    y1, y2 = ys.min().item(), ys.max().item()
    x1, x2 = xs.min().item(), xs.max().item()

    # Crop from *de-normalized* float image, convert to PIL
    crop_f = img_f[:, y1:y2+1, x1:x2+1].clamp(0,1)
    return to_pil_image(crop_f)


def main():
    os.makedirs(OUTDIR, exist_ok=True)

    # ----- fractioned test pairing -----
    pairs_all = paired_image_xml_list(IMG_DIR_TEST, XML_DIR_ALL)
    pairs     = subsample_pairs(pairs_all, fraction=F_TEST, seed=SEED)
    if len(pairs) == 0:
        raise RuntimeError("No test pairs found after fractioning. Check paths and F_TEST.")
    print(f"Using {len(pairs)}/{len(pairs_all)} test pairs (fraction={F_TEST})")

    # dataset & pick one sample
    ds = SquaresClassificationDatasetStream(pairs, canvas=CANVAS, train=False, use_padding_canvas=True)
    x, y = ds[0]  # take the first sample from the fractioned set

    # model + Grad-CAM
    model = build_resnet_classifier(num_classes=5)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()
    cammer = GradCAM(model, model.layer4[-1])

    # reconstruct the raw crop from the canvas tensor
    crop_pil = reconstruct_crop_from_canvas(x)

    # ---- Counterfactuals ----
    size_sweep(model, cammer, crop_pil, true_cls=y, canvas=CANVAS,
               scales=(0.6, 0.8, 1.0, 1.2, 1.4), outdir=OUTDIR)

    canvas_img = _paste_on_canvas(crop_pil, CANVAS)
    occlusion_study(model, cammer, canvas_img, true_cls=y, bar=16, where="top", outdir=OUTDIR)

    cammer.remove()
    print("Saved counterfactual plots to:", OUTDIR)

if __name__ == "__main__":
    main()
