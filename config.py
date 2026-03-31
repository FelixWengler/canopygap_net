# -----------------------
# Training Settings
# -----------------------

# Folder-based train/val inputs (expected subfolders: S2/ S1 and BDOM/)
TRAIN_ROOT = "/data/ahsoka/eocp/wengler/ground_fractions/bdom_grfra/chips/train"
VAL_ROOT   = "/data/ahsoka/eocp/wengler/ground_fractions/bdom_grfra/chips/val"

BATCH_SIZE = 4
EPOCHS = 60
LEARNING_RATE = 1e-4
NUM_BANDS = 10  # number of Sentinel-2 bands 
S1_BANDS = 3 # number of S1 bands

DEVICE = "cuda"  # or "cpu"

# Optional performance knobs
NUM_WORKERS = 4     # start with 0 or small number; increase if stable
NUM_THREADS = 28       # CPU threads for torch
AUGMENT = True         # if your dataset supports augment=True/False

# Output paths
MODEL_OUT = "/data/ahsoka/eocp/wengler/ground_fractions/S2height_grfra/models/output/S1S2_bdomgrfra_130326_2.pth"
LOG_PATH = "/data/ahsoka/eocp/wengler/ground_fractions/S2height_grfra/models/log/S1S2_bdomgrfra_130326_2.log"


# -----------------------
# Prediction Settings
# -----------------------
PREDICTION_INPUT = "/data/ahsoka/eocp/wengler/height_database/composite/median/0608/2025/2025_0608_median.tif"
PREDICTION_INPUT_S1_ALIGNED = "/data/ahsoka/eocp/wengler/height_database/S1/3035/res/S1_2025_VH_VV_stack_clip_res.tif"
PREDICTION_OUTPUT = "/data/ahsoka/eocp/wengler/ground_fractions/S2height_grfra/pred/bdom_grfra/2025bdom_S1S2_160326.tif"
PREDICTION_PATCH_SIZE = 256
PREDICTION_MODEL = "/data/ahsoka/eocp/wengler/ground_fractions/S2height_grfra/models/output/S1S2_bdomgrfra_130326_2_1381.pth"
PREDICTION_WORKERS = 30
PREDICTION_BATCH_SIZE = 8
