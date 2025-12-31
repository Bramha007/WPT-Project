# import os, torch, gc
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from captum.attr import IntegratedGradients, visualization as viz

# # Modular Imports from your provided files
# from src.setup import config_det as config
# from src.utils.device_utils import select_device
# from src.models.fasterrcnn import build_fasterrcnn_
# from src.dataio.det_dataset import GeometricShapeDataset, collate_fn
# from src.dataio.voc_parser import paired_image_xml_list
# from src.dataio.split_utils import subsample_pairs
# from src.dataio.det_transforms import Compose, ToTensor

# def run_proper_xai(limit=20):
#     # 0. GPU Cleanup: Resolve the OutOfMemoryError
#     gc.collect()
#     torch.cuda.empty_cache()

#     device = select_device(config.DEVICE)
#     torch.manual_seed(config.SEED) # Match the seed from train_shapes.py

#     xai_out_dir = os.path.join(config.OUTPUT_DIR, "xai_results")
#     os.makedirs(xai_out_dir, exist_ok=True)

#     # 1. Load Model with your exact latent_dim
#     num_classes = 2
#     model = build_fasterrcnn_(num_classes, config.LATENT_SIZE).to(device)
#     model.load_state_dict(torch.load(config.SAVE_CKPT, map_location=device))
#     model.eval()

#     # 2. Match the Data Split from train_shapes.py exactly
#     test_pairs_all = paired_image_xml_list(config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL_RECT)
#     test_pairs = subsample_pairs(test_pairs_all, config.F_TEST, seed=config.SEED)

#     ds = GeometricShapeDataset(test_pairs, transforms=Compose([ToTensor()]))
#     loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

#     # 3. Enhanced Wrapper: Ensure we explain high-confidence boxes
#     def wrapper_func(input_tensor):
#         outputs = model(list(input_tensor))
#         if len(outputs[0]['scores']) > 0:
#             # We explain the top-scoring box
#             return outputs[0]['scores'][0].view(1, 1)
#         return torch.zeros((1, 1), device=device)

#     ig = IntegratedGradients(wrapper_func)

#     print(f"Generating Matched XAI for {limit} images...")
#     for i, (imgs, tgts) in enumerate(tqdm(loader)):
#         if i >= limit: break

#         # FILENAME MATCHING: Extract ID to match pred_ID.png
#         img_path = test_pairs[i][0]
#         img_id = os.path.basename(img_path).split('.')[0].replace('img_', '')
#         save_name = f"xai_{img_id}.png"

#         input_img = imgs[0].to(device).unsqueeze(0).requires_grad_()

#         # 4. Use a WHITE baseline (all ones) for black geometric shapes
#         # This makes the "black" pixels of the quads much more significant to Captum
#         white_baseline = torch.ones_like(input_img).to(device)

#         # internal_batch_size=1 prevents the OutOfMemoryError on GPU
#         attr = ig.attribute(input_img, baselines=white_baseline, n_steps=50, internal_batch_size=1)

#         # Prepare for Visualization
#         attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
#         img_np = np.transpose(imgs[0].cpu().detach().numpy(), (1, 2, 0))

#         # 5. Visualizer call using the suggestion from your error log
#         fig, ax = viz.visualize_image_attr(
#             attr_np, img_np,
#             method="blended_heat_map",
#             sign="all",
#             show_colorbar=True,
#             title=f"XAI: {img_id}",
#             use_pyplot=False
#         )

#         fig.savefig(os.path.join(xai_out_dir, save_name))
#         plt.close(fig)

#         # Periodic memory cleanup
#         if i % 5 == 0: torch.cuda.empty_cache()

# if __name__ == "__main__":
#     run_proper_xai(limit=20)


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
        save_name = f"xai_{img_id}.png"

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

            # Prepare for Visualization
            # # We detach here because visualization doesn't need the graph
            # attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
            # img_np = np.transpose(imgs[0].cpu().detach().numpy(), (1, 2, 0))

            # Updated visualizer call as per Captum 0.6.0+
            # attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
            # img_np = np.transpose(imgs[0].cpu().detach().numpy(), (1, 2, 0))

            # max_val = np.max(np.abs(attr_np)) + 1e-9
            # attr_np_boosted = attr_np / max_val

            # img_np_light = img_np * 0.05 + 0.95
            # # attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
            # # img_np = np.transpose(imgs[0].cpu().detach().numpy(), (1, 2, 0))
            # # attr_np_boosted = attr_np / (np.max(np.abs(attr_np)) + 1e-9)

            # # fig, _ = viz.visualize_image_attr(
            # #     attr_np_boosted,
            # #     img_np,
            # #     method="blended_heat_map",
            # #     sign="all",
            # #     show_colorbar=True,
            # #     title=f"XAI for ID: {img_id}",
            # #     use_pyplot=False,
            # #     # Low alpha (0.1) makes the background shapes very light
            # #     alpha_overlay=0.1,
            # #     # outlier_perc=2 removes the top 2% of extreme values,
            # #     # which effectively brightens the rest of the heatmap
            # #     outlier_perc=2
            # # )
            # fig, _ = viz.visualize_image_attr(
            #     attr_np_boosted,
            #     img_np_light,
            #     method="blended_heat_map",
            #     sign="all",
            #     show_colorbar=True,
            #     title=f"High-Contrast XAI: {img_id}",
            #     use_pyplot=False,
            #     # Minimal alpha ensures attribution colors pop
            #     alpha_overlay=0.1,
            #     # Outlier clipping brightens the overall heat map
            #     outlier_perc=2
            # )
            # 5. Visualizer call with LIGHTER background
            # fig, _ = viz.visualize_image_attr(
            #     attr_np,
            #     img_np,
            #     method="blended_heat_map",
            #     sign="all",
            #     show_colorbar=True,
            #     title=f"XAI for ID: {img_id}",
            #     use_pyplot=False,
            #     # REDUCE alpha_overlay to make the background shapes lighter
            #     # 0.1 to 0.3 usually makes the heatmap colors "pop"
            #     alpha_overlay=0.2
            # )

            attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
            attr_np_boosted = np.sign(attr_np) * (np.abs(attr_np) ** 0.4)
            img_np_light = np.ones_like(attr_np_boosted) * 0.98
            fig, _ = viz.visualize_image_attr(
                attr_np_boosted,
                img_np_light,
                method="heat_map",
                sign="positive",
                cmap="seismic",
                show_colorbar=True,
                title=f"High-Visibility XAI: {img_id}",
                use_pyplot=False,
                outlier_perc=1,
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
