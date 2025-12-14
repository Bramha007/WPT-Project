# File: src/setup/config_cls.py (MULTI-TASK CONFIGURATION)

import os

# --- A. GENERAL SETUP & PATHS (Using Linux paths as default) ---


if os.name == 'nt':
    # Windows paths (for local testing)
    DATA_ROOT = r"E:\WPT-Project\Data\sized_squares_filled"
elif os.name == 'posix':
    DATA_ROOT = "/pitsec_sose2025_team3_1/data/sized_squares_filled" # for Linux
# DATA_ROOT = r"E:\WPT-Project\Data\sized_squares_filled"  # for Windows
# DATA_ROOT = "/pitsec_sose2025_team3_1/data/sized_squares_filled" # for Linux


# Output directory for classification results
OUTPUT_DIR = "outputs_v5_multi_task"
SAVE_CKPT = os.path.join(OUTPUT_DIR, "resnet_cls_multi_best.pt")

# Training, Validation, and Annotations paths
IMG_DIR_TRAIN = os.path.join(DATA_ROOT, "train")
IMG_DIR_VAL = os.path.join(DATA_ROOT, "val")
XML_DIR_ALL = os.path.join(DATA_ROOT, "annotations")

if os.name == 'nt':
    # Windows paths (for local testing)
    RECT_DATA_ROOT = r'E:\WPT-Project\Data\sized_rectangles_filled' # for Windows
elif os.name == 'posix':
    RECT_DATA_ROOT = "/pitsec_sose2025_team3_1/data/sized_rectangles_filled" # for Linux
# RECT_DATA_ROOT = r'E:\WPT-Project\Data\sized_rectangles_filled' # for Windows
# RECT_DATA_ROOT = "/pitsec_sose2025_team3_1/data/sized_rectangles_filled" # for Linux
IMG_DIR_TEST_RECT  = os.path.join(RECT_DATA_ROOT, 'test')
XML_DIR_ALL_RECT   = os.path.join(RECT_DATA_ROOT, 'annotations')

# --- B. MODEL & TRAINING PARAMETERS ---
EPOCHS = 4                 # Increased epochs for multi-task stability
BATCH_SIZE = 10              
LR = 1e-3
SEED = 42
NUM_WORKERS = 1              
DEVICE = "auto"
CANVAS_SIZE = 224
USE_PADDING_CANVAS = True
OPTIMIZER_NAME = "Adam"


# --- C. CLASSIFICATION CLASSES (TWO TASKS) ---

# TASK 1: AREA CLASSIFICATION (5 Classes, for Robustness)
AREA_BINS = [64, 256, 1024, 4096, 16384] 
NUM_CLS_AREA = len(AREA_BINS) 
AREA_NAMES = [str(a) for a in AREA_BINS]

# TASK 2: W/H REGRESSION (2 Outputs: W, H)
# The model will predict 2 continuous values.
NUM_REG_WH = 2

# --- D. DATA SUBSET FRACTIONS ---
F_TRAIN = 0.0005 
F_VAL   = 0.005
F_TEST  = 0.005
MAX_TRAIN_ITEMS = None