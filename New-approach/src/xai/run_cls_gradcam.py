import os, random
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.cls_dataset_stream import SquaresClassificationDatasetStream
from src.models.resnet_cls import build_resnet_classifier
from src.xai.gradcam_utils import GradCAM
from src.dataio.split_utils import subsample_pairs


CLASS_NAMES = ["8","16","32","64","128"]

def denorm(img):
    # img: [3,H,W] normalized by ImageNet stats
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    return (img * std + mean).clamp(0,1)

def overlay(img_hw, heatmap_hw, alpha=0.45):
    img = img_hw.permute(1,2,0).cpu().numpy()  # HWC
    hm  = heatmap_hw.squeeze(0).cpu().numpy()  # HxW
    hm  = plt.cm.jet(hm)[..., :3]              # colorize
    out = (1-alpha)*img + alpha*hm
    return np.clip(out, 0, 1)

def save_overlay(path, img, cam, title):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(3,3))
    plt.imshow(overlay(denorm(img), cam[0]))
    plt.axis("off")
    plt.title(title, fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()

def main():
    # --- paths & config
    DATA_ROOT = r"E:\WPT-Project\Data\sized_squares_filled"  # <-- edit if needed
    IMG_DIR_TEST = fr"{DATA_ROOT}\test"
    XML_DIR_ALL  = fr"{DATA_ROOT}\annotations"
    CANVAS = 224
    CKPT   = "outputs/resnet_cls.pt"
    OUTDIR = "outputs/xai_cls"
    N_PER_CLASS = 4

    # dataset & model
    test_pairs_all = paired_image_xml_list(IMG_DIR_TEST, XML_DIR_ALL)
    test_pairs     = subsample_pairs(test_pairs_all, fraction=0.05, seed=42)

    ds_test = SquaresClassificationDatasetStream(test_pairs, canvas=CANVAS, train=False, use_padding_canvas=True)

    model = build_resnet_classifier(num_classes=5)
    model.load_state_dict(torch.load(CKPT, map_location="cpu"))
    model.eval()

    # Grad-CAM hooked to last conv block
    cammer = GradCAM(model, model.layer4[-1])

    # collect a few samples per class
    per_class_idxs = {i: [] for i in range(5)}
    for i in range(len(ds_test)):
        _, y = ds_test[i]
        if len(per_class_idxs[y]) < N_PER_CLASS:
            per_class_idxs[y].append(i)
        if all(len(v) >= N_PER_CLASS for v in per_class_idxs.values()):
            break

    for cls_idx, idxs in per_class_idxs.items():
        for j, idx in enumerate(idxs):
            x, y = ds_test[idx]
            x = x.unsqueeze(0)
            cam, pred_cls, probs = cammer(x, target_class=None)
            p = float(probs[pred_cls].item())
            title = f"T:{CLASS_NAMES[y]}  P:{CLASS_NAMES[pred_cls]} ({p:.2f})"
            save_overlay(os.path.join(OUTDIR, f"gradcam_T{CLASS_NAMES[y]}_P{CLASS_NAMES[pred_cls]}_{j+1}.png"),
                         x[0], cam, title)

    cammer.remove()
    print("Saved Grad-CAM overlays to:", OUTDIR)

if __name__ == "__main__":
    main()
