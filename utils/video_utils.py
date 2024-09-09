import cv2

def read_video(video_path):
    """Reads video from video_path

    Args:
        video_path (str): Path of the video

    Returns:
        frames(list): A list that contains each frame of the input video
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def save_video(output_video_frames, output_video_path):
    """Saves video from frames

    Args:
        output_video_frames (list): Output frames to be saved in the video
        output_video_path (str): Path of the output video
    """
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (output_video_frames[0].shape[1], output_video_frames[0].shape[0]))
    for frame in output_video_frames:
        out.write(frame)
    out.release()