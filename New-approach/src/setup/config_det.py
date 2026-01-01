# File: config.py

import os

# ====================================================================
# A. GENERAL SETUP & PATHS
# ====================================================================

if os.name == "nt":
    # Windows paths (for local testing)
    DATA_ROOT = r"E:\WPT-Project\Data\sized_squares_filled"
elif os.name == "posix":
    DATA_ROOT = "/pitsec_sose2025_team3_1/data/sized_squares_filled"  # for Linux
# DATA_ROOT = r"E:\WPT-Project\Data\sized_squares_filled"  # for Windows
# DATA_ROOT = "/pitsec_sose2025_team3_1/data/sized_squares_filled" # for Linux


# Output directory for quadrilateral detection results
# Task-level folder
TASK_DIR = "quad_detection"

# Variation-level folder (The 3 folders for latents)
LATENT_SIZE = None  # Change to 256, 512, or 1024 for other runs
OUTPUT_DIR = os.path.join(TASK_DIR, f"latent_{LATENT_SIZE}")

# File-specific paths
SAVE_CKPT = os.path.join(OUTPUT_DIR, "fasterrcnn_best.pt")
DET_SUMMARY = os.path.join(OUTPUT_DIR, "det_run_summary.json")

# Training, Validation, and Annotations paths
IMG_DIR_TRAIN = os.path.join(DATA_ROOT, "train")
IMG_DIR_VAL = os.path.join(DATA_ROOT, "val")
XML_DIR_ALL = os.path.join(DATA_ROOT, "annotations")

if os.name == "nt":
    # Windows paths (for local testing)
    RECT_DATA_ROOT = r"E:\WPT-Project\Data\sized_rectangles_filled"  # for Windows
elif os.name == "posix":
    RECT_DATA_ROOT = (
        "/pitsec_sose2025_team3_1/data/sized_rectangles_filled"  # for Linux
    )
# RECT_DATA_ROOT = r'E:\WPT-Project\Data\sized_rectangles_filled' # for Windows
# RECT_DATA_ROOT = "/pitsec_sose2025_team3_1/data/sized_rectangles_filled" # for Linux
IMG_DIR_TEST_RECT = os.path.join(RECT_DATA_ROOT, "test")
XML_DIR_ALL_RECT = os.path.join(RECT_DATA_ROOT, "annotations")

# ====================================================================
# B. MODEL & TRAINING PARAMETERS
# ====================================================================

EPOCHS = 10  # Increased for better training
BATCH_SIZE = 10  # Increased for GPU efficiency
LR = 0.001  # Learning Rate
SEED = 42  # Ensures deterministic results
NUM_WORKERS = 1  # Use 4-8 workers when on GPU to prevent data bottlenecks

# DEVICE: 'auto' checks for CUDA first, then defaults to CPU.
# Use 'cpu' to force CPU, or 'cuda' to force GPU.
DEVICE = "auto"

OPTIMIZER_NAME = "SGD"  # "AdamW" # Options: "SGD", "AdamW"

# LATENT_SIZE = (
#     512  # Size of the latent vector in the Faster R-CNN box head eg 128, 512, 1024
# )

# ====================================================================
# C. DATA SUBSET FRACTIONS
# ====================================================================

# Fractions (0.0 to 1.0) to subsample the dataset splits
F_TRAIN = 0.0005*10
F_VAL = 0.002
F_TEST = 0.002

# F_TRAIN = 0.1
# F_VAL = 0.1
# F_TEST = 0.1

# Hard-cap the training items if the fraction still yields too much data
MAX_TRAIN_ITEMS = None
