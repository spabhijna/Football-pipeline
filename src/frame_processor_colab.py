"""
Improved frame processor for Colab to reduce video flickering
"""

from typing import Any, Optional
import numpy as np
import supervision as sv
from ultralytics import YOLO
from src.mini_map import create_mini_map, overlay_mini_map
from src.utils import resolve_goalkeepers_team_id
from src.team import TeamClassifier, TeamConsistencyTracker
from src.config import BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID

# Global variables for frame processing state
PLAYER_DETECTION_MODEL = None
KEYPOINT_MODEL = None
tracker = None
team_classifier = None
team_tracker = None
ellipse_annotator = None
label_annotator = None
triangle_annotator = None

# Anti-flickering state variables
previous_team_assignments = {}
frame_count = 0
stable_assignments_threshold = 5

# Performance optimization variables
team_prediction_cache = {}  # Cache predictions to avoid recomputation
cache_max_size = 100
last_cache_clear = 0
classification_frequency = 10  # Classify every N frames for performance


def process_frame_stable(frame: np.ndarray, frame_idx: int) -> np.ndarray:
    """
    Process a single frame with anti-flickering improvements for Colab.
    
    Args:
        frame: Input video frame
        frame_idx: Frame index for tracking
        
    Returns:
        Annotated frame with stable detections and tracking
    """
    global previous_team_assignments, frame_count, team_prediction_cache, last_cache_clear
    frame_count = frame_idx
    
    try:
        # Run inference for player detection with slightly higher confidence
        result = PLAYER_DETECTION_MODEL.predict(frame, conf=0.4, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Process ball detections
        ball_detections = detections[detections.class_id == BALL_ID]
        ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

        # Process other detections with improved NMS
        all_detections = detections[detections.class_id != BALL_ID]
        all_detections = all_detections.with_nms(threshold=0.3, class_agnostic=True)  # Stricter NMS
        all_detections = tracker.update_with_detections(detections=all_detections)

        # Separate different object types
        goalkeepers_detections = all_detections[all_detections.class_id == GOALKEEPER_ID]
        players_detections = all_detections[all_detections.class_id == PLAYER_ID]
        referees_detections = all_detections[all_detections.class_id == REFEREE_ID]

        # Optimized team assignment with GPU support and caching
        if len(players_detections) > 0:
            players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
            
            # Performance optimization: reduce team classification frequency
            should_classify = (
                frame_idx == 0 or  # Always classify first frame
                frame_idx % classification_frequency == 0 or  # Classify every Nth frame
                len(previous_team_assignments) < len(players_detections)  # New players detected
            )
            
            try:
                # Check if team classifier is properly fitted
                if (hasattr(team_classifier, 'cluster_model') and 
                    hasattr(team_classifier.cluster_model, 'cluster_centers_') and
                    should_classify):
                    
                    # Use caching to avoid repeated computations
                    cache_key = f"{len(players_crops)}_{frame_idx//classification_frequency}"
                    
                    if cache_key in team_prediction_cache:
                        current_team_predictions = team_prediction_cache[cache_key]
                        if frame_idx == 0:
                            print("✅ Team classifier ready, using AI-based classification")
                    else:
                        # GPU-optimized prediction with smaller batch size
                        if hasattr(team_classifier, 'batch_size'):
                            original_batch_size = team_classifier.batch_size
                            team_classifier.batch_size = min(8, len(players_crops))  # Smaller batches
                        
                        try:
                            current_team_predictions = team_classifier.predict(players_crops)
                            
                            # Cache the result
                            if len(team_prediction_cache) < cache_max_size:
                                team_prediction_cache[cache_key] = current_team_predictions
                            
                            if frame_idx == 0:
                                print("✅ Team classifier ready, using AI-based classification")
                                
                        except Exception as pred_error:
                            if frame_idx == 0:
                                print(f"⚠️ GPU prediction failed, using spatial fallback: {pred_error}")
                            raise AttributeError("Prediction failed")
                        finally:
                            # Restore original batch size
                            if hasattr(team_classifier, 'batch_size'):
                                team_classifier.batch_size = original_batch_size
                else:
                    # Use previous predictions if classifier not ready or skipping frame
                    if not should_classify and previous_team_assignments:
                        # Use cached assignments for non-classification frames
                        current_team_predictions = [
                            previous_team_assignments.get(tid, 0) 
                            for tid in players_detections.tracker_id
                        ]
                    else:
                        raise AttributeError("Team classifier not fitted or needs spatial fallback")
                
                # Apply stability tracking with fallback
                stabilized_team_ids = []
                for i, tracker_id in enumerate(players_detections.tracker_id):
                    if tracker_id is not None:
                        if i < len(current_team_predictions):
                            # Use team tracker for consistency
                            stabilized_team = team_tracker.update_team_assignment(
                                tracker_id, current_team_predictions[i]
                            )
                            
                            # Additional stability check: avoid rapid changes
                            if tracker_id in previous_team_assignments:
                                if frame_idx < stable_assignments_threshold:
                                    # Keep previous assignment for very early frames
                                    stabilized_team = previous_team_assignments[tracker_id]
                                elif abs(stabilized_team - previous_team_assignments[tracker_id]) > 0:
                                    # Team changed - apply additional verification
                                    if frame_idx % classification_frequency != 0:  # Only allow changes on classification frames
                                        stabilized_team = previous_team_assignments[tracker_id]
                            
                            previous_team_assignments[tracker_id] = stabilized_team
                            stabilized_team_ids.append(stabilized_team)
                        else:
                            # Fallback for index out of range
                            prev_team = previous_team_assignments.get(tracker_id, 0)
                            stabilized_team_ids.append(prev_team)
                    else:
                        # No tracker ID - use spatial assignment
                        center_x = (players_detections.xyxy[i][0] + players_detections.xyxy[i][2]) / 2
                        team_id = 0 if center_x < frame.shape[1] / 2 else 1
                        stabilized_team_ids.append(team_id)

                players_detections.class_id = np.array(stabilized_team_ids, dtype=int)
                
            except (AttributeError, ValueError, Exception) as e:
                # Improved fallback: use spatial-based team assignment
                if frame_idx == 0 or frame_idx % 100 == 0:
                    print(f"Using spatial assignment (frame {frame_idx}): {type(e).__name__}")
                
                team_ids = []
                frame_width = frame.shape[1]
                for i, xyxy in enumerate(players_detections.xyxy):
                    # Spatial-based team assignment with tracker consistency
                    center_x = (xyxy[0] + xyxy[2]) / 2
                    tracker_id = players_detections.tracker_id[i] if i < len(players_detections.tracker_id) else None
                    
                    # Use previous assignment if available for consistency
                    if tracker_id is not None and tracker_id in previous_team_assignments:
                        team_id = previous_team_assignments[tracker_id]
                    else:
                        # Spatial assignment: left side = team 0, right side = team 1
                        team_id = 0 if center_x < frame_width / 2 else 1
                        if tracker_id is not None:
                            previous_team_assignments[tracker_id] = team_id
                    
                    team_ids.append(team_id)
                    
                players_detections.class_id = np.array(team_ids, dtype=int)
            
            # Clear cache periodically to prevent memory buildup
            global last_cache_clear
            if frame_idx - last_cache_clear > 100:
                team_prediction_cache.clear()
                last_cache_clear = frame_idx

        # Stabilize goalkeeper teams
        if len(goalkeepers_detections) > 0:
            goalkeepers_detections.class_id = resolve_goalkeepers_team_id(
                players_detections, goalkeepers_detections
            )

        # Handle referees
        if len(referees_detections) > 0:
            referees_detections.class_id = np.full(len(referees_detections), 2, dtype=int)

        # Merge all detections
        all_detections = sv.Detections.merge(
            [players_detections, goalkeepers_detections, referees_detections]
        )

        # Create stable labels
        labels = []
        for tracker_id in all_detections.tracker_id:
            if tracker_id is not None:
                labels.append(f"#{tracker_id}")
            else:
                labels.append("?")

        # Ensure class_id is integer
        all_detections.class_id = all_detections.class_id.astype(int)

        # Annotate main frame
        annotated_frame = frame.copy()
        
        # Apply annotations with error handling
        try:
            annotated_frame = ellipse_annotator.annotate(
                scene=annotated_frame, detections=all_detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=all_detections, labels=labels
            )
            annotated_frame = triangle_annotator.annotate(
                scene=annotated_frame, detections=ball_detections
            )
        except Exception as e:
            print(f"Annotation error at frame {frame_idx}: {e}")
            # Continue with basic frame if annotation fails

        # Create and overlay mini map with error handling
        try:
            mini_map = create_mini_map(
                frame,
                ball_detections,
                players_detections,
                referees_detections,
                goalkeepers_detections,
                KEYPOINT_MODEL,
            )
            if mini_map is not None:
                annotated_frame = overlay_mini_map(
                    annotated_frame, mini_map, position="bottom_right", scale=0.2
                )
        except Exception as e:
            print(f"Mini-map error at frame {frame_idx}: {e}")
            # Continue without mini-map if it fails

        return annotated_frame

    except Exception as e:
        print(f"Critical error processing frame {frame_idx}: {e}")
        return frame  # Return original frame on critical error


def initialize_frame_processor_stable(
    player_detection_model: YOLO,
    keypoint_model: YOLO,
    tracker_obj: sv.ByteTrack,
    team_classifier_obj: TeamClassifier,
    team_tracker_obj: TeamConsistencyTracker,
    ellipse_ann: sv.EllipseAnnotator,
    label_ann: sv.LabelAnnotator,
    triangle_ann: sv.TriangleAnnotator,
    ball_id: int = BALL_ID,
    goalkeeper_id: int = GOALKEEPER_ID,
    player_id: int = PLAYER_ID,
    referee_id: int = REFEREE_ID,
    use_gpu_for_detection: bool = True,
    use_gpu_for_team_classifier: bool = False,
) -> None:
    """
    Initialize all global variables for stable frame processing in Colab.
    
    Args:
        use_gpu_for_detection: Use GPU for YOLO models (recommended: True)
        use_gpu_for_team_classifier: Use GPU for team classifier (recommended: False for stability)
    """
    global PLAYER_DETECTION_MODEL, KEYPOINT_MODEL, tracker, team_classifier, team_tracker
    global ellipse_annotator, label_annotator, triangle_annotator
    global BALL_ID, GOALKEEPER_ID, PLAYER_ID, REFEREE_ID
    global previous_team_assignments, frame_count
    
    import torch
    
    # Configure YOLO models for GPU/CPU
    if use_gpu_for_detection and torch.cuda.is_available():
        print("🚀 Using GPU for YOLO detection models")
        player_detection_model.to('cuda')
        keypoint_model.to('cuda')
    else:
        print("💻 Using CPU for YOLO detection models")
        player_detection_model.to('cpu')
        keypoint_model.to('cpu')
    
    # Team classifier device management
    if hasattr(team_classifier_obj, 'device'):
        if use_gpu_for_team_classifier and torch.cuda.is_available():
            print("🚀 Using GPU for team classifier")
            if hasattr(team_classifier_obj, 'features_model'):
                team_classifier_obj.features_model.to('cuda')
                team_classifier_obj.device = 'cuda'
                # Optimize batch size for GPU
                team_classifier_obj.batch_size = min(team_classifier_obj.batch_size, 16)
        else:
            print("💻 Using CPU for team classifier (recommended for stability)")
            if hasattr(team_classifier_obj, 'features_model'):
                team_classifier_obj.features_model.to('cpu')
                team_classifier_obj.device = 'cpu'
                # Smaller batch size for CPU
                team_classifier_obj.batch_size = min(team_classifier_obj.batch_size, 8)
    
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
    
    # Reset anti-flickering state
    previous_team_assignments = {}
    frame_count = 0
    
    # Reset performance optimization variables
    global team_prediction_cache, last_cache_clear, classification_frequency
    team_prediction_cache = {}
    last_cache_clear = 0
    classification_frequency = 5 if use_gpu_for_team_classifier else 10  # More frequent if using GPU
    
    # Memory optimization for GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")t 
    
    print("✅ Stable frame processor initialized for Colab")