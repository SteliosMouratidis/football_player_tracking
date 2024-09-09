⚽ Player Collision Detection and Tracking
This project is designed to detect and track players in a soccer match video, identify potential collisions using Intersection over Union (IoU), and annotate these events on the video. The project utilizes a YOLO model trained specifically for player detection, and the collision detection is based on calculating the IoU between player bounding boxes.

🚀 Features
Player Detection: Uses a YOLO model to detect players in each frame of the video.
Tracking: Tracks players across frames using the supervision library and ByteTrack.
Collision Detection: Identifies collisions between players using IoU.
Annotation: Annotates detected players and collisions on the video.
🛠️ Requirements
Before running this project, ensure you have the following installed:

1. Ultralytics:
pip install ultralytics
2. OpenCV:
pip install opencv-python
3. Supervision:
pip install supervision
4. Roboflow:
pip install roboflow

📦 Dependencies
OpenCV: For video processing and frame manipulation.
YOLO: Using the ultralytics library for YOLO model handling.
Supervision: For tracking and detection management.
Torch: For handling the YOLO model.


📊 Dataset
The YOLO model used in this project is trained on the GoalVisionPlayers Dataset provided by Roboflow Universe. This dataset includes images of soccer players annotated with bounding boxes, used to train the YOLO model for accurate player detection. (https://universe.roboflow.com/bs-3hjo9/goalvisionplayers/dataset/7)

Run the Script:

Open main.py in VSCode.
Modify the video paths in main.py if necessary.

Review Collision Coordinates:
The (x, y) coordinates of all detected collisions are printed to the terminal.

🗂️ Project Structure
main.py: Main script to run the video processing.
trackers.py: Contains the Tracker class for handling detection, tracking, and collision detection.
utils: Utility functions for video processing and bounding box calculations.