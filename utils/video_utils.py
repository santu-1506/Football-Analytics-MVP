# Video I/O helpers — read all frames from a video, save frames as output video

import cv2

def read_video(video_path):
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
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        output_video_path,
        fourcc,
        24,                                            # fps
        (output_video_frames[0].shape[1],              # width
         output_video_frames[0].shape[0])              # height
    )
    for frame in output_video_frames:
        out.write(frame)
    out.release()

# Function Does read_video(path) Reads every frame into a list — entire video lives in memory as numpy arrays save_video(frames, path) Writes annotated frames back out as an .avi file at 24fps

# One thing to know — read_video loads the full video into RAM. For a short match clip (30–60 seconds) this is totally fine. For a full 90 minute match, you'd need to process it in chunks — but for this MVP, don't worry about it.