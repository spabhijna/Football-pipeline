import numpy as np
import supervision as sv
from src.mini_map import create_mini_map, overlay_mini_map
from src.utils import resolve_goalkeepers_team_id

# Global variables that need to be set before using process_frame
PLAYER_DETECTION_MODEL = None
KEYPOINT_MODEL = None
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3

# Global objects that need to be initialized
tracker = None
team_classifier = None
team_tracker = None
ellipse_annotator = None
label_annotator = None
triangle_annotator = None


def process_frame(frame: np.ndarray, _) -> np.ndarray:
    """
    Process a single frame for football analysis with player detection, tracking, and mini-map overlay.

    Args:
        frame: Input video frame
        _: Unused parameter (for compatibility with video processing functions)

    Returns:
        Annotated frame with detections, tracking, and mini-map overlay
    """
    # Run inference for player detection
    result = PLAYER_DETECTION_MODEL.predict(frame, conf=0.3, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    # Process ball detections
    ball_detections = detections[detections.class_id == BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    # Process other detections
    all_detections = detections[detections.class_id != BALL_ID]
    all_detections = all_detections.with_nms(threshold=0.5, class_agnostic=True)
    all_detections = tracker.update_with_detections(detections=all_detections)

    # Separate different object types
    goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
    players_detections = all_detections[all_detections.class_id == PLAYER_ID]
    referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

    # Team assignment with stabilization
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    if players_crops:
        current_team_predictions = team_classifier.predict(players_crops)
        # Apply team consistency tracking
        stabilized_team_ids = []
        for i, tracker_id in enumerate(players_detections.tracker_id):
            if tracker_id is not None:
                stabilized_team = team_tracker.update_team_assignment(
                    tracker_id, current_team_predictions[i]
                )
                stabilized_team_ids.append(stabilized_team)
            else:
                stabilized_team_ids.append(current_team_predictions[i])

        players_detections.class_id = np.array(stabilized_team_ids)

    # Stabilize goalkeeper teams - assign team IDs (0 or 1) to goalkeepers
    goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
        players_detections, goalkeepers_detections
    )
    
    # Create mini map BEFORE modifying referee class_ids
    # At this point: players have team IDs (0/1), goalkeepers have team IDs (0/1)
    mini_map = create_mini_map(
        frame,
        ball_detections,
        players_detections,
        referees_detections,
        goalkeepers_detections,
        KEYPOINT_MODEL
    )

    # Now adjust referee class_id for display purposes
    referees_detections.class_id -= 1

    all_detections = sv.Detections.merge(
        [players_detections, goalkeepers_detections, referees_detections]
    )

    # Create labels
    labels = [
        f"#{tracker_id}" 
        for tracker_id in 
        all_detections.tracker_id
        ]

    all_detections.class_id = all_detections.class_id.astype(int)

    # Annotate main frame
    annotated_frame = frame.copy()
    annotated_frame = ellipse_annotator.annotate(
        scene=annotated_frame, detections=all_detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=all_detections, labels=labels
    )
    annotated_frame = triangle_annotator.annotate(
        scene=annotated_frame, detections=ball_detections
    )

    # Overlay mini map (already created before referee class_id adjustment)
    if mini_map is not None:
        annotated_frame = overlay_mini_map(
            annotated_frame, mini_map, position="bottom_right", scale=0.2
        )

    return annotated_frame


def initialize_frame_processor(
    player_detection_model,
    keypoint_model,
    tracker_obj,
    team_classifier_obj,
    team_tracker_obj,
    ellipse_ann,
    label_ann,
    triangle_ann,
    ball_id=0,
    goalkeeper_id=1,
    player_id=2,
    referee_id=3,
):
    """
    Initialize all global variables needed for frame processing.

    Args:
        player_detection_model: YOLO model for player detection
        keypoint_model: YOLO model for keypoint detection
        tracker_obj: ByteTracker object for tracking
        team_classifier_obj: TeamClassifier object
        team_tracker_obj: Team tracking object for stabilization
        ellipse_ann: EllipseAnnotator for players
        label_ann: LabelAnnotator for labels
        triangle_ann: TriangleAnnotator for ball
        ball_id, goalkeeper_id, player_id, referee_id: Class IDs for different objects
    """
    global \
        PLAYER_DETECTION_MODEL, \
        KEYPOINT_MODEL, \
        tracker, \
        team_classifier, \
        team_tracker
    global ellipse_annotator, label_annotator, triangle_annotator
    global BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID

    PLAYER_DETECTION_MODEL = player_detection_model
    KEYPOINT_MODEL = keypoint_model
    tracker = tracker_obj
    team_classifier = team_classifier_obj
    team_tracker = team_tracker_obj
    ellipse_annotator = ellipse_ann
    label_annotator = label_ann
    triangle_annotator = triangle_ann
    BALL_ID = ball_id
    GOALKEEPER_ID = goalkeeper_id
    PLAYER_ID = player_id
    REFEREE_ID = referee_id
