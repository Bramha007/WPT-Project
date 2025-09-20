import os, json, glob
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from captum.attr import IntegratedGradients, Saliency

from src.models.resnet_cls import build_resnet_classifier
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.split_utils import subsample_pairs

# -------- config --------
DATA_ROOT    = r"E:\WPT-Project\Data\sized_squares_filled"
IMG_DIR_TEST = fr"{DATA_ROOT}\test"
XML_DIR_ALL  = fr"{DATA_ROOT}\annotations"

CKPT_CLS     = "outputs/resnet_cls.pt"
OUT_DIR      = "outputs/xai_cls_captum"

F_TEST       = 0.02      # small fraction for speed
K_IMAGES     = 12        # limit visualizations
TARGET_CLASS = None      # set to 0..4 to force class; None = use model's argmax
STEPS_IG     = 64        # steps for Integrated Gradients
# ------------------------

def tfm_eval(pil):
    import torchvision.transforms as T
    return T.Compose([
        T.Resize((224,224)),
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
    out = (1-alpha)*img + alpha*cm
    return np.clip(out, 0, 1)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cpu")

    # model
    model = build_resnet_classifier(num_classes=5).to(device)
    model.load_state_dict(torch.load(CKPT_CLS, map_location=device))
    model.eval()

    # data subset
    pairs_all = paired_image_xml_list(IMG_DIR_TEST, XML_DIR_ALL)
    pairs = subsample_pairs(pairs_all, fraction=F_TEST, seed=42)[:K_IMAGES]

    ig = IntegratedGradients(model)
    sal = Saliency(model)

    for img_path, _ in pairs:
        stem = Path(img_path).stem
        pil = Image.open(img_path).convert("RGB")
        x = tfm_eval(pil).unsqueeze(0).to(device)  # [1,3,224,224]

        with torch.no_grad():
            logits = model(x)
            pred = int(logits.argmax(1).item())
        target = pred if TARGET_CLASS is None else TARGET_CLASS

        # Integrated Gradients
        x.requires_grad_(True)
        baseline = torch.zeros_like(x)  # black baseline (also try blurred/avg if you want)
        atts_ig = ig.attribute(x, baselines=baseline, target=target, n_steps=STEPS_IG)
        atts_ig = atts_ig.squeeze(0).abs().sum(0)      # [H,W] channel-sum
        atts_ig = (atts_ig - atts_ig.min()) / (atts_ig.max() - atts_ig.min() + 1e-6)

        # Saliency (vanilla gradient)
        x.grad = None
        x.requires_grad_(True)
        atts_sal = sal.attribute(x, target=target)
        atts_sal = atts_sal.squeeze(0).abs().sum(0)
        atts_sal = (atts_sal - atts_sal.min()) / (atts_sal.max() - atts_sal.min() + 1e-6)

        vis = denorm(x[0].detach())
        over_ig  = overlay(vis, atts_ig, alpha=0.5)
        over_sal = overlay(vis, atts_sal, alpha=0.5)

        # save
        plt.imsave(os.path.join(OUT_DIR, f"{stem}_ig_cls{target}.png"), over_ig)
        plt.imsave(os.path.join(OUT_DIR, f"{stem}_saliency_cls{target}.png"), over_sal)
        print(f"Saved Captum maps for {stem} (target={target})")

    print("Done. Captum (classifier) results in", OUT_DIR)

if __name__ == "__main__":
    main()
