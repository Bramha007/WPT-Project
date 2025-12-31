import os, torch, gc
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


def run_proper_xai(limit=20):
    # 0. GPU Cleanup
    gc.collect()
    torch.cuda.empty_cache()

    device = select_device(config.DEVICE)
    torch.manual_seed(config.SEED)

    xai_out_dir = os.path.join(config.OUTPUT_DIR, "xai_results")
    os.makedirs(xai_out_dir, exist_ok=True)

    # 1. Load Model
    num_classes = 2
    model = build_fasterrcnn_(num_classes, config.LATENT_SIZE).to(device)
    model.load_state_dict(torch.load(config.SAVE_CKPT, map_location=device))

    # CRITICAL: Ensure model remains in eval mode but allows gradient computation
    model.eval()

    # 2. Match Data Split
    test_pairs_all = paired_image_xml_list(
        config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL_RECT
    )
    test_pairs = subsample_pairs(test_pairs_all, config.F_TEST, seed=config.SEED)

    ds = GeometricShapeDataset(test_pairs, transforms=Compose([ToTensor()]))
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # 3. Fixed Wrapper: Preserving the Gradient Connection
    def wrapper_func(input_tensor):
        # Captum passes a batch tensor; we convert to list for Faster R-CNN
        # Ensure gradients are tracked during this forward pass
        outputs = model(list(input_tensor))

        # Explain the most confident detection
        if len(outputs[0]["scores"]) > 0:
            # We take the score directly without .item() or .detach()
            # We use .view(1, 1) to ensure it is a 2D tensor for Captum
            return outputs[0]["scores"][0].view(1, 1)

        # If no detection, return a 0 tensor that still requires grad
        return torch.tensor([[0.0]], device=device, requires_grad=True)

    ig = IntegratedGradients(wrapper_func)

    print(f"Generating Matched XAI for {limit} images...")
    for i, (imgs, tgts) in enumerate(tqdm(loader)):
        if i >= limit:
            break

        img_path = test_pairs[i][0]
        img_id = os.path.basename(img_path).split(".")[0].replace("img_", "")

        # CRITICAL: input_img must require_grad BEFORE reaching IntegratedGradients
        input_img = imgs[0].to(device).unsqueeze(0)
        input_img.requires_grad = True

        # Use White baseline for black quads on white background
        white_baseline = torch.ones_like(input_img).to(device)

        try:
            # internal_batch_size=1 is crucial for GPUs with limited memory
            attr = ig.attribute(
                input_img, baselines=white_baseline, n_steps=50, internal_batch_size=1
            )

            # attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
            # attr_np_boosted = np.sign(attr_np) * (np.abs(attr_np) ** 0.4)
            # img_np_light = np.ones_like(attr_np_boosted) * 0.98
            # attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
            # # Absolute values for thresholding
            # abs_attr = np.abs(attr_np)
            # # Keep only strong attributions (top 10%)
            # threshold = np.percentile(abs_attr, 90)
            # # Zero out weak attributions → neutral background
            # attr_np_clean = np.where(abs_attr >= threshold, attr_np, 0.0)

            # fig, _ = viz.visualize_image_attr(
            #     attr_np_clean,
            #     img_np_light,
            #     method="heat_map",
            #     sign="positive",
            #     cmap="seismic",
            #     show_colorbar=True,
            #     title=f"Clean XAI: {img_id}",
            #     use_pyplot=False,
            # )
            # --- 1. PREPARE DATA ---
            attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
            img_np = np.transpose(imgs[0].cpu().detach().numpy(), (1, 2, 0))

            # --- 2. BOOST AND CLEAN ATTRIBUTIONS ---
            abs_attr = np.abs(attr_np)
            threshold = np.percentile(abs_attr, 90) # Keep top 10%
            attr_np_clean = np.where(abs_attr >= threshold, attr_np, 0.0)
            
            # Normalize to ensure maximum color saturation
            attr_np_clean = attr_np_clean / (np.max(np.abs(attr_np_clean)) + 1e-9)

            # --- 3. CREATE LIGHT GHOST BACKGROUND ---
            # Instead of a flat constant, scale the real image to [0.9, 1.0]
            # This turns black shapes into a light gray (0.9) and white stays white (1.0)
            img_np_light = img_np * 0.1 + 0.9 

            # --- 4. RENDER ---
            fig, _ = viz.visualize_image_attr(
                attr_np_clean,
                img_np_light,
                method="blended_heat_map", # Blended helps colors stand out
                sign="all",                # Show both Positive (Green) and Negative (Red)
                show_colorbar=True,
                title=f"High-Contrast XAI: {img_id}",
                use_pyplot=False,
                alpha_overlay=0.1,         # Make the ghost background very faint
                outlier_perc=2             # Brightens the heatmap colors
            )

            fig.savefig(os.path.join(xai_out_dir, f"xai_hc_{img_id}.png"))
            plt.close(fig)

        except Exception as e:
            print(f"Skipping image {img_id} due to error: {e}")

        # Periodic memory cleanup
        if i % 2 == 0:
            torch.cuda.empty_cache()
            gc.collect()


if __name__ == "__main__":
    run_proper_xai(limit=20)
