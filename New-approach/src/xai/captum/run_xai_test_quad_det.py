# import os, torch, gc
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from captum.attr import IntegratedGradients, visualization as viz

# # Modular Imports
# from src.setup import config_det as config
# from src.utils.device_utils import select_device
# from src.models.fasterrcnn import build_fasterrcnn_
# from src.dataio.det_dataset import GeometricShapeDataset, collate_fn
# from src.dataio.voc_parser import paired_image_xml_list
# from src.dataio.split_utils import subsample_pairs
# from src.dataio.det_transforms import Compose, ToTensor


# def run_proper_xai(limit=20):
#     # 0. GPU Cleanup
#     gc.collect()
#     torch.cuda.empty_cache()

#     device = select_device(config.DEVICE)
#     torch.manual_seed(config.SEED)

#     xai_out_dir = os.path.join(config.OUTPUT_DIR, "xai_results")
#     os.makedirs(xai_out_dir, exist_ok=True)

#     # 1. Load Model
#     num_classes = 2
#     model = build_fasterrcnn_(num_classes, config.LATENT_SIZE).to(device)
#     model.load_state_dict(torch.load(config.SAVE_CKPT, map_location=device))

#     # CRITICAL: Ensure model remains in eval mode but allows gradient computation
#     model.eval()

#     # 2. Match Data Split
#     test_pairs_all = paired_image_xml_list(
#         config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL_RECT
#     )
#     test_pairs = subsample_pairs(test_pairs_all, config.F_TEST, seed=config.SEED)

#     ds = GeometricShapeDataset(test_pairs, transforms=Compose([ToTensor()]))
#     loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     # 3. Fixed Wrapper: Preserving the Gradient Connection
#     # def wrapper_func(input_tensor):
#     #     # Captum passes a batch tensor; we convert to list for Faster R-CNN
#     #     # Ensure gradients are tracked during this forward pass
#     #     outputs = model(list(input_tensor))

#     #     # Explain the most confident detection
#     #     if len(outputs[0]["scores"]) > 0:
#     #         # We take the score directly without .item() or .detach()
#     #         # We use .view(1, 1) to ensure it is a 2D tensor for Captum
#     #         return outputs[0]["scores"][0].view(1, 1)

#     #     # If no detection, return a 0 tensor that still requires grad
#     #     return torch.tensor([[0.0]], device=device, requires_grad=True)

#     def wrapper_func(input_tensor, target_idx=0):
#         outputs = model(list(input_tensor))
#         if len(outputs[0]['scores']) > target_idx:
#             # Target a specific detection score
#             return outputs[0]['scores'][target_idx].view(1, 1)
#         return torch.tensor([[0.0]], device=device, requires_grad=True)

#     ig = IntegratedGradients(wrapper_func)

#     print(f"Generating Matched XAI for {limit} images...")
#     # for i, (imgs, tgts) in enumerate(tqdm(loader)):
#     #     if i >= limit:
#     #         break

#     #     img_path = test_pairs[i][0]
#     #     img_id = os.path.basename(img_path).split(".")[0].replace("img_", "")

#     #     # CRITICAL: input_img must require_grad BEFORE reaching IntegratedGradients
#     #     input_img = imgs[0].to(device).unsqueeze(0)
#     #     input_img.requires_grad = True

#     #     # Use White baseline for black quads on white background
#     #     white_baseline = torch.ones_like(input_img).to(device)

#     # try:
#     #         # internal_batch_size=1 is crucial for GPUs with limited memory
#     #         attr = ig.attribute(
#     #             input_img, baselines=white_baseline, n_steps=50, internal_batch_size=1
#     #         )

#     #         # attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
#     #         # attr_np_boosted = np.sign(attr_np) * (np.abs(attr_np) ** 0.4)
#     #         # img_np_light = np.ones_like(attr_np_boosted) * 0.98
#     #         # attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
#     #         # # Absolute values for thresholding
#     #         # abs_attr = np.abs(attr_np)
#     #         # # Keep only strong attributions (top 10%)
#     #         # threshold = np.percentile(abs_attr, 90)
#     #         # # Zero out weak attributions → neutral background
#     #         # attr_np_clean = np.where(abs_attr >= threshold, attr_np, 0.0)

#     #         # fig, _ = viz.visualize_image_attr(
#     #         #     attr_np_clean,
#     #         #     img_np_light,
#     #         #     method="heat_map",
#     #         #     sign="positive",
#     #         #     cmap="seismic",
#     #         #     show_colorbar=True,
#     #         #     title=f"Clean XAI: {img_id}",
#     #         #     use_pyplot=False,
#     #         # )
#     #         # --- 1. PREPARE DATA ---
#     #         attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
#     #         img_np = np.transpose(imgs[0].cpu().detach().numpy(), (1, 2, 0))

#     #         # --- 2. BOOST AND CLEAN ATTRIBUTIONS ---
#     #         abs_attr = np.abs(attr_np)
#     #         threshold = np.percentile(abs_attr, 90) # Keep top 10%
#     #         attr_np_clean = np.where(abs_attr >= threshold, attr_np, 0.0)
            
#     #         # Normalize to ensure maximum color saturation
#     #         attr_np_clean = attr_np_clean / (np.max(np.abs(attr_np_clean)) + 1e-9)

#     #         # --- 3. CREATE LIGHT GHOST BACKGROUND ---
#     #         # Instead of a flat constant, scale the real image to [0.9, 1.0]
#     #         # This turns black shapes into a light gray (0.9) and white stays white (1.0)
#     #         img_np_light = img_np * 0.1 + 0.9 

#     #         # --- 4. RENDER ---
#     #         fig, _ = viz.visualize_image_attr(
#     #             attr_np_clean,
#     #             img_np_light,
#     #             method="heat_map", # Blended helps colors stand out
#     #             sign="all",                # Show both Positive (Green) and Negative (Red)
#     #             show_colorbar=True,
#     #             title=f"High-Contrast XAI: {img_id}",
#     #             use_pyplot=False,
#     #             alpha_overlay=0.1,         # Make the ghost background very faint
#     #             outlier_perc=2             # Brightens the heatmap colors
#     #         )

#     #         fig.savefig(os.path.join(xai_out_dir, f"xai_hc_{img_id}.png"))
#     #         plt.close(fig)

#     for i, (imgs, tgts) in enumerate(tqdm(loader)):
#         if i >= limit: break
        
#         img_path = test_pairs[i][0]
#         img_id = os.path.basename(img_path).split(".")[0].replace("img_", "")
        
#         input_img = imgs[0].to(device).unsqueeze(0)
#         input_img.requires_grad = True
#         white_baseline = torch.ones_like(input_img).to(device)

#         # First, run a forward pass to see how many quads were detected
#         with torch.no_grad():
#             full_output = model([input_img.squeeze(0)])[0]
#             scores = full_output['scores']
#             boxes = full_output['boxes']

#         # Loop through every detection above threshold
#         for det_idx, score in enumerate(scores):
#             if score < 0.7: continue # Ignore low-confidence noise
            
#             # Generate XAI for this specific quad
#             attr = ig.attribute(
#                 input_img, 
#                 baselines=white_baseline, 
#                 target=None, # We handle target in the wrapper_func via det_idx
#                 additional_forward_args=(det_idx,), 
#                 n_steps=50, 
#                 internal_batch_size=1
#             )

#             # --- Visualization Logic ---
#             attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
#             img_np = np.transpose(imgs[0].cpu().detach().numpy(), (1, 2, 0))
            
#             # Contrast boosting
#             abs_attr = np.abs(attr_np)
#             threshold = np.percentile(abs_attr, 90)
#             attr_np_clean = np.where(abs_attr >= threshold, attr_np, 0.0)
#             img_np_light = img_np * 0.1 + 0.9 

#             fig, _ = viz.visualize_image_attr(
#                 attr_np_clean, img_np_light, method="blended_heat_map",
#                 sign="all", show_colorbar=True, alpha_overlay=0.1,
#                 title=f"XAI ID: {img_id} | Quad: {det_idx} (Score: {score:.2f})",
#                 use_pyplot=False
#             )
            
#             # Save each quad as a separate file
#             fig.savefig(os.path.join(xai_out_dir, f"xai_{img_id}_quad_{det_idx}.png"))
#             plt.close(fig)

#         # except Exception as e:
#         #     print(f"Skipping image {img_id} due to error: {e}")

#         # Periodic memory cleanup
#     if i % 2 == 0:
#         torch.cuda.empty_cache()
#         gc.collect()


# if __name__ == "__main__":
#     run_proper_xai(limit=20)



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


def run_proper_xai(limit=20):
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

        # 🔥 aggregate all detections
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
    run_proper_xai(limit=50)
