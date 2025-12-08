import os
import sys
from pathlib import Path
import supervision as sv
from datetime import datetime

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import get_models
from src.team import TeamClassifier, TeamConsistencyTracker
from src.frame_processor import initialize_frame_processor, process_frame
from src.annotators import get_annotators, get_tracker

# Configuration
INPUT_VIDEO_PATH = "../inputs/test_input2.mp4"
OUTPUT_DIR = "../outputs"
MODELS_DIR = "../models"

# Model paths
PLAYER_DETECTION_MODEL_PATH = f"{MODELS_DIR}/player-detection.pt"
KEYPOINT_MODEL_PATH = f"{MODELS_DIR}/Keypoint-model-new.pt"

# Detection class IDs
BALL_ID = 0
GOALKEEPER_ID = 1
PLAYER_ID = 2
REFEREE_ID = 3


def create_output_directory():
    """Create output directory with timestamp"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(OUTPUT_DIR) / f"analysis_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def setup_models_and_annotators():
    """Initialize all models and annotators"""
    print("Loading models...")

    # Load detection models
    player_detection_model, keypoint_model = get_models(
        player_detection_model_path=PLAYER_DETECTION_MODEL_PATH,
        keypoint_model_path=KEYPOINT_MODEL_PATH,
    )

    # Initialize team classifier
    team_classifier = TeamClassifier(device="cpu")

    # Initialize tracker
    tracker = get_tracker()
    tracker.reset()

    # Initialize team tracker
    team_tracker = TeamConsistencyTracker()
    ellipse_annotator, label_annotator, triangle_annotator = get_annotators()
    # Setup annotators

    print("Models and annotators loaded successfully!")

    return (
        player_detection_model,
        keypoint_model,
        team_classifier,
        tracker,
        team_tracker,
        ellipse_annotator,
        label_annotator,
        triangle_annotator,
    )


def collect_initial_crops(
    input_path: str, player_detection_model, num_frames: int = 10
):
    """
    Collect player crops from initial frames to train the team classifier.

    Args:
        input_path: Path to input video
        player_detection_model: YOLO model for player detection
        num_frames: Number of frames to sample

    Returns:
        List of player crops
    """
    crops = []
    frame_generator = sv.get_video_frames_generator(input_path)

    for frame_idx, frame in enumerate(frame_generator):
        if frame_idx >= num_frames:
            break

        # Run inference
        result = player_detection_model.predict(frame, conf=0.3, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Get player detections
        players_detections = detections[detections.class_id == PLAYER_ID]

        # Crop players
        for xyxy in players_detections.xyxy:
            crop = sv.crop_image(frame, xyxy)
            crops.append(crop)

    return crops


def process_video(input_path: str, output_path: Path):
    """Process the entire video and save output"""

    # Setup models and annotators
    (
        player_detection_model,
        keypoint_model,
        team_classifier,
        tracker,
        team_tracker,
        ellipse_annotator,
        label_annotator,
        triangle_annotator,
    ) = setup_models_and_annotators()

    print(f"Processing video: {input_path}")

    # Get video info
    video_info = sv.VideoInfo.from_video_path(input_path)
    print(
        f"Video info: {video_info.width}x{video_info.height}, {video_info.fps}fps, {video_info.total_frames} frames"
    )

    # Collect initial crops and train team classifier
    print("Collecting initial player crops for team classification...")
    initial_crops = collect_initial_crops(
        input_path, player_detection_model, num_frames=10
    )

    if len(initial_crops) < 2:
        print(
            "Warning: Not enough players detected in initial frames. Team classification may not work properly."
        )
    else:
        print(f"Training team classifier on {len(initial_crops)} player crops...")
        team_classifier.fit(initial_crops)
        print("Team classifier trained!")

    # Initialize frame processor
    initialize_frame_processor(
        player_detection_model=player_detection_model,
        keypoint_model=keypoint_model,
        tracker_obj=tracker,
        team_classifier_obj=team_classifier,
        team_tracker_obj=team_tracker,
        ellipse_ann=ellipse_annotator,
        label_ann=label_annotator,
        triangle_ann=triangle_annotator,
        ball_id=BALL_ID,
        goalkeeper_id=GOALKEEPER_ID,
        player_id=PLAYER_ID,
        referee_id=REFEREE_ID,
    )

    # Create output video path
    output_video_path = output_path / f"analyzed_{Path(input_path).name}"

    # Process video
    with sv.VideoSink(str(output_video_path), video_info) as sink:
        frame_generator = sv.get_video_frames_generator(input_path)

        for frame_idx, frame in enumerate(frame_generator):
            # Process frame
            processed_frame = process_frame(frame, frame_idx)

            # Write processed frame
            sink.write_frame(processed_frame)

            # Progress update
            if frame_idx % 30 == 0:  # Every second at 30fps
                progress = (frame_idx + 1) / video_info.total_frames * 100
                print(
                    f"Progress: {progress:.1f}% ({frame_idx + 1}/{video_info.total_frames} frames)"
                )

    print(f"Video processing completed! Output saved to: {output_video_path}")
    return output_video_path


def main():
    """Main function to run the video analysis pipeline"""
    print("=== Football Video Analysis Pipeline ===")
    print(f"Input video: {INPUT_VIDEO_PATH}")

    # Check if input video exists
    if not os.path.exists(INPUT_VIDEO_PATH):
        print(f"Error: Input video not found at {INPUT_VIDEO_PATH}")
        return

    # Create output directory
    output_dir = create_output_directory()
    print(f"Output directory: {output_dir}")

    try:
        # Process the video
        output_video_path = process_video(INPUT_VIDEO_PATH, output_dir)

        print("\=== Processing Complete ===")
        print(f"Processed video: {output_video_path}")
        print(f"Output directory: {output_dir}")
        print("\nFeatures applied:")
        print("Player detection and tracking")
        print("Ball detection")
        print("Team classification")
        print("Tactical mini-map overlay")
        print("Real-time annotations")

    except Exception as e:
        print(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
