from typing import Tuple
from ultralytics import YOLO


def get_models(
    player_detection_model_path: str = "models/best.pt",
    keypoint_model_path: str = "models/keypoint_model.pt",
) -> Tuple[YOLO, YOLO]:
    PLAYER_DETECTION_MODEL = YOLO(player_detection_model_path)
    KEYPOINT_MODEL = YOLO(keypoint_model_path)

    return PLAYER_DETECTION_MODEL, KEYPOINT_MODEL
