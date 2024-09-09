import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import pickle
import os
from utils import get_center_of_bbox, get_bbox_width

class Tracker:
    def __init__(self, model_path: str, batch_size: int = 20, conf_threshold: float = 0.1, collision_threshold: float = 0.7):
        """
        Initialization of the Tracker class

        Args:
             model_path (str): Path of the YOLO .pt model.
            batch_size (int, optional): Batch size for the model prediction. Defaults to 20.
            conf_threshold (float, optional): Confidence threshold for the model prediction. Defaults to 0.1.
            collision_threshold (float, optional): IoU threshold to control the sensitivity of the collision detection. Defaults to 0.5.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        self.batch_size = batch_size
        self.conf_threshold = conf_threshold
        self.collision_threshold = collision_threshold
        self.collision_count = 0  

    def detect_frames(self, frames: list) -> list:
        """
        Run the model prediction to detection player in frames

        Args:
            frames (list): List of frames from the video.

        Returns:
            list: A list of detections predicted by the model for each frame.
        """
        detections = []
        for i in range(0, len(frames), self.batch_size):
            detections_batch = self.model.predict(frames[i:i+self.batch_size], conf=self.conf_threshold)
            detections += detections_batch
        return detections

    def get_object_tracks(self, frames: list, read_from_stub: bool = False, stub_path: str = None) -> dict:
        """
        Performs object tracking based on the supervision library (https://supervision.roboflow.com/latest/)

        Args:
            frames (list): List of frames from the video.
            read_from_stub (bool, optional): Whether to read tracking data from a pre-saved stub to improve speed. Defaults to False.
            stub_path (str, optional): Path of the saved stub (as pickle). Defaults to None.

        Returns:
            dict: A dictionary containing all tracked objects, including their bounding boxes and IDs.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        
        # Initialize tracks dictionary
        tracks = {
            "players": [],
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert to supervision detection format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Track objects
            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

            tracks["players"].append({})
            players_bboxes = []

            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                track_id = frame_detection[4]

                if cls_id == cls_names_inv['players']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    players_bboxes.append((bbox, track_id))

            # Check for collisions between players
            self.detect_collisions(players_bboxes, frame_num, frames[frame_num])


        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def detect_collisions(self, players_bboxes, frame_num, frame):
        """
        Detect collisions between players, draw them, and print collision details.

        Args:
            players_bboxes (list): List of tuples containing player bounding boxes and their track IDs.
            frame_num (int): The current frame number being processed.
            frame (numpy.ndarray): The current frame from the video.
        """
        for i in range(len(players_bboxes)):
            for j in range(i + 1, len(players_bboxes)):
                box1, track_id1 = players_bboxes[i]
                box2, track_id2 = players_bboxes[j]
                if self.iou(box1, box2) > self.collision_threshold:
                    # Collision detected, increment the counter
                    self.collision_count += 1
                    
                    # Calculate the center of the collision
                    collision_x = int((box1[0] + box1[2]) / 2)
                    collision_y = int((box1[1] + box1[3]) / 2)
                    
                    # Draw the collision point on the frame
                    self.draw_collision(frame, collision_x, collision_y)
                    
                    # Print the collision details
                    print(f"Collision detected in frame {frame_num}:")
                    print(f"  Location: ({collision_x}, {collision_y})")
                    print(f"  Players involved: ID {track_id1} and ID {track_id2}")

    def iou(self, box1, box2):
        """
       Calculate Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1 (list): Coordinates of the first bounding box [x1, y1, x2, y2].
            box2 (list): Coordinates of the second bounding box [x1, y1, x2, y2].

        Returns:
            float: The IoU value between the two bounding boxes.
        """
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    def draw_collision(self, frame, x, y):
        """
         Draw a red dot at the collision point on the frame.

        Args:
            frame (numpy.ndarray): The current frame from the video.
            x (int): X-coordinate of the collision point.
            y (int): Y-coordinate of the collision point.
        """
        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

    def draw_ellipse(self, frame, bbox, color, track_id):
        """
         Draw an ellipse around the detected player on the frame.

        Args:
            frame (numpy.ndarray): The current frame from the video.
            bbox (list): Bounding box coordinates [x1, y1, x2, y2] for the player.
            color (tuple): Color of the ellipse in BGR format.
            track_id (int): ID of the tracked player.

        Returns:
            numpy.ndarray: The frame with the ellipse drawn around the player.
        """
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame, 
            center=(x_center, y2), 
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=45,
            endAngle=235, 
            color=color, 
            thickness=2, 
            lineType=cv2.LINE_4
        )
        return frame

    def draw_annotations(self, video_frames: list, tracks: dict) -> list:
        """
        Draw annotations on video frames based on tracking data.

        Args:
            video_frames (list): List of frames from the video.
            tracks (dict): Dictionary containing tracking data for players.

        Returns:
            list: List of video frames with annotations (ellipses and collision points) drawn on them.
        """
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"]
            
            for track_id, player in player_dict[frame_num].items():
                frame = self.draw_ellipse(frame, player["bbox"], (255, 0, 0), track_id)

            output_video_frames.append(frame)

        return output_video_frames