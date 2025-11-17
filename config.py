import os


class Config:
    # Path to SemanticKITTI sequences
    SEMANTICKITTI_ROOT = "/mnt/d/Subhransu workspace/Dataset/my_kitti_dataset/dataset/sequences"

    # Output root for preprocessed data
    RMOS_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "rmos")

    # Voxel size (meters)
    VOXEL_SIZE = 0.05

    # Training
    NUM_CLASSES = 20  # adjust to your mapping if needed
    BATCH_SIZE = 4
    NUM_WORKERS = 4
    MAX_ITER = 40000
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    PRINT_FREQ = 20
    VAL_FREQ = 1000

    # RMOSUNet channels
    BASE_CHANNELS = 48
