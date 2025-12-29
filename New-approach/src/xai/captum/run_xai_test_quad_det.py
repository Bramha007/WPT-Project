# import os
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# from torch.utils.data import DataLoader
# from captum.attr import IntegratedGradients, visualization as viz

# from src.setup import config_det as config
# from src.utils.device_utils import select_device 
# from src.models.fasterrcnn import build_fasterrcnn_
# from src.dataio.det_dataset import GeometricShapeDataset, collate_fn
# from src.dataio.voc_parser import paired_image_xml_list
# from src.dataio.split_utils import subsample_pairs
# from src.dataio.det_transforms import Compose, ToTensor

# def run_xai_on_test_set(limit=None):
#     device = select_device(config.DEVICE)
#     torch.manual_seed(config.SEED)
#     xai_out_dir = os.path.join(config.OUTPUT_DIR, "xai_results")
#     os.makedirs(xai_out_dir, exist_ok=True)

#     # 1. Load Model
#     num_classes = 2
#     model = build_fasterrcnn_(num_classes, config.LATENT_SIZE).to(device)
#     model.load_state_dict(torch.load(config.SAVE_CKPT, map_location=device))
#     model.eval()

#     # 2. Prepare Data
    
#     test_pairs_all = paired_image_xml_list(
#         config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL_RECT
#     )
#     test_pairs = subsample_pairs(test_pairs_all, config.F_TEST, seed=config.SEED)


#     ds = GeometricShapeDataset(test_pairs, transforms=Compose([ToTensor()]))
#     # loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)
#     loader =  DataLoader(
#         ds,
#         batch_size=config.BATCH_SIZE,
#         # shuffle=train and len(pairs) > 1,
#         shuffle=False,
#         num_workers=config.NUM_WORKERS,
#         collate_fn=collate_fn,
#         # pin_memory=is_cuda,  # Pin memory dramatically speeds up host-to-device transfers
#     )

#     # 3. Define Wrapper for Captum
#     def wrapper_func(input_tensor):
#         outputs = model(list(input_tensor))
#         if len(outputs[0]['scores']) > 0:
#             # Reshape to [1, 1] to satisfy Captum's batch requirements
#             return outputs[0]['scores'][0].view(1, 1) 
#         return torch.zeros((1, 1), device=device)

#     ig = IntegratedGradients(wrapper_func)

#     # 4. Loop and Save Explanations
#     print(f"Generating XAI maps for {len(ds) if limit is None else limit} images...")
#     for i, (imgs, tgts) in enumerate(tqdm(loader)):
#         if limit and i >= limit: break
        
#         # Prepare input with gradients enabled
#         input_img = imgs[0].to(device).unsqueeze(0).requires_grad_()
        
#         # FIXED: Removed target=0 and the extra call outside the loop
#         attr = ig.attribute(input_img, n_steps=50, internal_batch_size=1)
        
#         # Prepare for Visualization
#         attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
#         img_np = np.transpose(imgs[0].cpu().detach().numpy(), (1, 2, 0))

#         # Create the Heatmap Overlay
#         fig, ax = viz.visualize_image_attr(
#             attr_np, 
#             img_np, 
#             method="blended_heat_map", 
#             sign="all", 
#             show_colorbar=True,
#             use_pyplot=False  # Returns fig instead of calling plt.show()
#         )
        
#         # Save results
#         save_path = os.path.join(xai_out_dir, f"xai_test_{i}.png")
#         fig.savefig(save_path)
#         plt.close(fig)

# if __name__ == "__main__":
#     run_xai_on_test_set(limit=10)

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients, visualization as viz

from src.setup import config_det as config
from src.utils.device_utils import select_device 
from src.models.fasterrcnn import build_fasterrcnn_
from src.dataio.det_dataset import GeometricShapeDataset, collate_fn
from src.dataio.voc_parser import paired_image_xml_list
from src.dataio.split_utils import subsample_pairs
from src.dataio.det_transforms import Compose, ToTensor

def run_xai_on_test_set(limit=None):
    device = select_device(config.DEVICE)
    torch.manual_seed(config.SEED)
    xai_out_dir = os.path.join(config.OUTPUT_DIR, "xai_results")
    os.makedirs(xai_out_dir, exist_ok=True)

    # 1. Load Model
    num_classes = 2
    model = build_fasterrcnn_(num_classes, config.LATENT_SIZE).to(device)
    model.load_state_dict(torch.load(config.SAVE_CKPT, map_location=device))
    model.eval()

    # 2. Prepare Data (Ensuring match with test model)
    test_pairs_all = paired_image_xml_list(
        config.IMG_DIR_TEST_RECT, config.XML_DIR_ALL_RECT
    )
    # Identical seed and fraction as your test model evaluation
    test_pairs = subsample_pairs(test_pairs_all, config.F_TEST, seed=config.SEED)

    ds = GeometricShapeDataset(test_pairs, transforms=Compose([ToTensor()]))
    
    # Set batch_size=1 for XAI to simplify mapping to filenames
    loader = DataLoader(
        ds,
        batch_size=1, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn,
    )

    # 3. Define Wrapper for Captum
    def wrapper_func(input_tensor):
        outputs = model(list(input_tensor))
        if len(outputs[0]['scores']) > 0:
            # Reshape to [1, 1] to satisfy Captum's batch requirements
            return outputs[0]['scores'][0].view(1, 1) 
        return torch.zeros((1, 1), device=device)

    ig = IntegratedGradients(wrapper_func)

    # 4. Loop and Save Explanations
    print(f"Generating Matched XAI maps...")
    for i, (imgs, tgts) in enumerate(tqdm(loader)):
        if limit and i >= limit: break
        
        # Get actual filename and prepend 'xai_'
        img_path = test_pairs[i][0]
        base_filename = os.path.basename(img_path).split('.')[0]
        save_name = f"xai_{base_filename}.png"

        # Prepare input with gradients enabled
        input_img = imgs[0].to(device).unsqueeze(0).requires_grad_()
        
        # Calculate attribution
        attr = ig.attribute(input_img, n_steps=50, internal_batch_size=1)
        
        # Prepare for Visualization
        attr_np = np.transpose(attr.squeeze().cpu().detach().numpy(), (1, 2, 0))
        img_np = np.transpose(imgs[0].cpu().detach().numpy(), (1, 2, 0))

        # Create the Heatmap Overlay
        fig, ax = viz.visualize_image_attr(
            attr_np, 
            img_np, 
            method="blended_heat_map", 
            sign="all", 
            show_colorbar=True,
            title=f"XAI for {base_filename}",
            use_pyplot=False  
        )
        
        # Save results with prepended name
        save_path = os.path.join(xai_out_dir, save_name)
        fig.savefig(save_path)
        plt.close(fig)

if __name__ == "__main__":
    run_xai_on_test_set(limit=10)