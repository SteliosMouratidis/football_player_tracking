from trackers import Tracker
import cv2
from utils.video_utils import save_video

def process_video(video_path, output_path, tracker):
    """
    Processes a video to detect and track players, annotates collisions, and saves the processed video.

    Args:
        video_path (str): Path to the input video file.
        output_path (str): Path where the annotated output video will be saved.
        tracker (Tracker): Instance of the Tracker class used for object tracking and collision detection.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize the video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame with tracking and collision detection
        tracks = tracker.get_object_tracks([frame])
        annotated_frame = tracker.draw_annotations([frame], tracks)[0]

        # Write the annotated frame to the output video
        out.write(annotated_frame)

    cap.release()
    out.release()

    print(f"Processed video saved as {output_path}")
    print(f"Total number of collisions detected: {tracker.collision_count}")

def main():
    """
    Main function to initialize the Tracker and process the video.
    
    The Tracker is initialized with a YOLO model for player detection. The video is processed 
    frame by frame to detect and track players, and collisions are annotated in the output video.
    """
    # Initialize Tracker
    tracker = Tracker(
        model_path='models/best_yolo_goalvision.pt'
        )

    # Process the video frame by frame
    process_video('OHL_20220227-175500_OHL_W_vs_Charleroi_OHL-PANO-A1_short.mp4', 'output_videos/full_video_processed.avi', tracker)

if __name__ == '__main__':
    main()