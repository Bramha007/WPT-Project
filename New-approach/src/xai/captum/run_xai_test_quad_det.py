import os, gc, torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients, visualization as viz

# Modular Imports
from src.setup import config_det as config
from src.utils.device_utils import select_device 
from src.models.fasterrcnn import build_fasterrcnn_
from src.dataio.det_dataset import GeometricShapeDataset, collate_fn
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.split_utils import subsample_pairs
from src.dataio.det_transforms import Compose, ToTensor


def run_proper_xai(limit):
    # --------------------------------------------------
    # 0. Cleanup
    # --------------------------------------------------
    gc.collect()
    torch.cuda.empty_cache()

    device = select_device(config.DEVICE)
    torch.manual_seed(config.SEED)

    out_dir = os.path.join(config.OUTPUT_DIR, "xai_results")
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------
    # 1. Load model
    # --------------------------------------------------
    model = build_fasterrcnn_(num_classes=2, latent_dim=config.LATENT_SIZE)
    model.load_state_dict(torch.load(config.SAVE_CKPT, map_location=device))
    model.to(device)
    model.eval()

    # --------------------------------------------------
    # 2. Dataset (same split as training)
    # --------------------------------------------------
    test_pairs_all = paired_image_xml_list(
        config.IMG_DIR_TEST_RECT,
        config.XML_DIR_ALL_RECT
    )
    test_pairs = subsample_pairs(
        test_pairs_all,
        config.F_TEST,
        seed=config.SEED
    )

    ds = GeometricShapeDataset(test_pairs, transforms=Compose([ToTensor()]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # --------------------------------------------------
    # 3. Wrapper: explain ALL detected quads
    # --------------------------------------------------
    def wrapper_func(input_tensor):
        outputs = model(list(input_tensor))
        scores = outputs[0]["scores"]

        if scores.numel() == 0:
            return torch.tensor([[0.0]], device=device, requires_grad=True)

        score_thresh = 0.3
        valid_scores = scores[scores >= score_thresh]

        if valid_scores.numel() == 0:
            return torch.tensor([[0.0]], device=device, requires_grad=True)

        # aggregate all detections
        return valid_scores.sum().view(1, 1)

    ig = IntegratedGradients(wrapper_func)

    # --------------------------------------------------
    # 4. XAI loop
    # --------------------------------------------------
    print(f"Generating XAI for {limit} images...")

    for i, (imgs, _) in enumerate(tqdm(loader)):
        if i >= limit:
            break

        img_path = test_pairs[i][0]
        img_id = os.path.basename(img_path).split(".")[0]
        save_path = os.path.join(out_dir, f"xai_all_{img_id}.png")

        input_img = imgs[0].unsqueeze(0).to(device)
        input_img.requires_grad = True

        white_baseline = torch.ones_like(input_img).to(device)

        try:
            torch.cuda.empty_cache()
            # IG
            attr = ig.attribute(
                input_img,
                baselines=white_baseline,
                n_steps=50,
                internal_batch_size=1
            )

            # --------------------------------------------------
            # 5. Post-processing for clean visualization
            # --------------------------------------------------
            attr_np = np.transpose(
                attr.squeeze().cpu().detach().numpy(),
                (1, 2, 0)
            )

            # Power-law boost
            attr_np = np.sign(attr_np) * (np.abs(attr_np) ** 0.4)

            # Remove weak attribution → neutral background
            abs_attr = np.abs(attr_np)
            thresh = np.percentile(abs_attr, 90)
            attr_np[abs_attr < thresh] = 0.0

            # Pure neutral background
            bg = np.ones_like(attr_np) * 0.98

            # --------------------------------------------------
            # 6. Visualize
            # --------------------------------------------------
            fig, ax = viz.visualize_image_attr(
                attr_np,
                bg,
                method="heat_map",
                sign="positive",
                cmap="seismic",
                show_colorbar=True,
                title=f"XAI (all detections): {img_id}",
                use_pyplot=False
            )

            # --------------------------------------------------
            # 7. Overlay detected bounding boxes
            # --------------------------------------------------
            with torch.no_grad():
                outputs = model(list(input_img))

            boxes = outputs[0]["boxes"].cpu().numpy()
            scores = outputs[0]["scores"].cpu().numpy()

            for box, score in zip(boxes, scores):
                if score < 0.3:
                    continue
                x1, y1, x2, y2 = box
                ax.add_patch(
                    plt.Rectangle(
                        (x1, y1),
                        x2 - x1,
                        y2 - y1,
                        fill=False,
                        edgecolor="black",
                        linewidth=2
                    )
                )

            fig.savefig(save_path, bbox_inches="tight", dpi=200)
            plt.close(fig)

        except Exception as e:
            print(f"Skipping {img_id}: {e}")

        if i % 2 == 0:
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    run_proper_xai(limit=100)
