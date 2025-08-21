# Path Directories
DATASET_DIR = "datasets"
CHECKPOINTS_DIR = "lightning_logs"

# Reproducability
STREAM_SHUFFLE_BUFFER_SIZE = 5000 # WARNING: changing this will change the test set across runs; not recommended to change 
SEED = 42