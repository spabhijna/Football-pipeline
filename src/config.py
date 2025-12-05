# Detection class IDs
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# File paths
INPUT_VIDEO_PATH = "../inputs/121364_0.mp4"
OUTPUT_DIR = "../outputs"
MODELS_DIR = "../models"

# Model paths
PLAYER_DETECTION_MODEL_PATH = f"{MODELS_DIR}/player-detection.pt"
KEYPOINT_MODEL_PATH = f"{MODELS_DIR}/keypoint-detection.pt"

# Team model configuration
SIGLIP_MODEL_PATH = "google/siglip-base-patch16-224"

# Visualization constants
PITCH_PADDING = 50
PITCH_SCALE = 0.1
PITCH_LINE_THICKNESS = 4
PITCH_POINT_RADIUS = 8
PLAYER_POINT_RADIUS = 10
PLAYER_POINT_THICKNESS = 2
PATH_THICKNESS = 2
BALL_POINT_RADIUS = 10

# Mini map constants
MINI_MAP_POSITION = "bottom_right"
MINI_MAP_SCALE = 0.2
MINI_MAP_BORDER_THICKNESS = 2
