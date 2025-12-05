import numpy as np
import cv2
import supervision as sv
from src.soccer import draw_pitch, draw_points_on_pitch
from src.soccer_pitch import SoccerPitchConfiguration
from src.view import ViewTransformer

# Initialize configuration
CONFIG = SoccerPitchConfiguration()


def create_mini_map(
    frame,
    ball_detections,
    players_detections,
    referees_detections,
    goalkeepers_detections,
    keypoint_model,
):
    """Create mini map with pitch and player positions"""
    try:
        # Detect pitch key points
        result = keypoint_model.predict(frame, conf=0.3, verbose=False)[0]
        key_points = sv.KeyPoints.from_ultralytics(result)

        # Project ball, players and referees on pitch
        filter = key_points.confidence[0] > 0.5
        if np.sum(filter) < 4:
            return None

        frame_reference_points = key_points.xy[0][filter]
        pitch_reference_points = np.array(CONFIG.vertices)[filter]

        transformer = ViewTransformer(
            source=frame_reference_points, target=pitch_reference_points
        )

        # Transform positions to pitch coordinates
        frame_ball_xy = ball_detections.get_anchors_coordinates(
            sv.Position.BOTTOM_CENTER
        )
        pitch_ball_xy = (
            transformer.transform_points(points=frame_ball_xy)
            if len(frame_ball_xy) > 0
            else np.array([])
        )

        all_players_detections = sv.Detections.merge(
            [players_detections, goalkeepers_detections]
        )
        players_xy = all_players_detections.get_anchors_coordinates(
            sv.Position.BOTTOM_CENTER
        )
        pitch_players_xy = (
            transformer.transform_points(points=players_xy)
            if len(players_xy) > 0
            else np.array([])
        )

        referees_xy = referees_detections.get_anchors_coordinates(
            sv.Position.BOTTOM_CENTER
        )
        pitch_referees_xy = (
            transformer.transform_points(points=referees_xy)
            if len(referees_xy) > 0
            else np.array([])
        )

        # Create mini map
        mini_map = draw_pitch(CONFIG)

        # Draw players (team 0 - blue)
        if len(pitch_players_xy) > 0:
            team_0_mask = all_players_detections.class_id == 0
            if np.any(team_0_mask):
                mini_map = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_players_xy[team_0_mask],
                    face_color=sv.Color.from_hex("00BFFF"),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=mini_map,
                )

        # Draw players (team 1 - pink)
        if len(pitch_players_xy) > 0:
            team_1_mask = all_players_detections.class_id == 1
            if np.any(team_1_mask):
                mini_map = draw_points_on_pitch(
                    config=CONFIG,
                    xy=pitch_players_xy[team_1_mask],
                    face_color=sv.Color.from_hex("FF1493"),
                    edge_color=sv.Color.BLACK,
                    radius=16,
                    pitch=mini_map,
                )

        # Draw referees (yellow)
        if len(pitch_referees_xy) > 0:
            mini_map = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_referees_xy,
                face_color=sv.Color.from_hex("FFD700"),
                edge_color=sv.Color.BLACK,
                radius=16,
                pitch=mini_map,
            )

        # Draw ball (white)
        if len(pitch_ball_xy) > 0:
            mini_map = draw_points_on_pitch(
                config=CONFIG,
                xy=pitch_ball_xy,
                face_color=sv.Color.WHITE,
                edge_color=sv.Color.BLACK,
                radius=10,
                pitch=mini_map,
            )

        return mini_map
    except Exception as e:
        print(f"Error creating mini map: {e}")
        return None


def overlay_mini_map(main_frame, mini_map, position="bottom_right", scale=0.2):
    """Overlay mini map on the main frame

    Args:
        main_frame: Main video frame to overlay the mini map on
        mini_map: Mini map to overlay (tactical view)
        position: Position of overlay ('bottom_right', 'bottom_left', 'top_right', 'top_left')
        scale: Scale factor for mini map size relative to main frame width

    Returns:
        main_frame with mini map overlaid
    """
    if mini_map is None:
        return main_frame

    h_main, w_main = main_frame.shape[:2]
    h_mini, w_mini = mini_map.shape[:2]

    # Calculate scaled dimensions
    new_width = int(w_main * scale)
    new_height = int((h_mini / w_mini) * new_width)

    # Resize mini map
    mini_map_resized = cv2.resize(mini_map, (new_width, new_height))

    # Calculate position
    if position == "bottom_right":
        x_offset = w_main - new_width - 10
        y_offset = h_main - new_height - 10
    elif position == "bottom_left":
        x_offset = 10
        y_offset = h_main - new_height - 10
    elif position == "top_right":
        x_offset = w_main - new_width - 10
        y_offset = 10
    else:  # top_left
        x_offset = 10
        y_offset = 10

    # Ensure coordinates are within bounds
    x_offset = max(0, min(x_offset, w_main - new_width))
    y_offset = max(0, min(y_offset, h_main - new_height))

    # Overlay mini map
    main_frame[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = (
        mini_map_resized
    )

    # Add border
    border_thickness = 2
    cv2.rectangle(
        main_frame,
        (x_offset, y_offset),
        (x_offset + new_width, y_offset + new_height),
        (255, 255, 255),
        border_thickness,
    )

    return main_frame
