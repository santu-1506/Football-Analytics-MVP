# Dev/test script — use this to quickly test individual modules without running the full pipeline

import cv2
from utils import read_video, save_video
from tracker import Tracker


# ── Quick test — change these paths as needed ──────────────────────────────────
VIDEO_PATH  = 'input_videos/sample.mp4'
OUTPUT_PATH = 'output_videos/test_output.avi'
MODEL_PATH  = 'yolo11l.pt'
STUB_PATH   = 'stubs/track_stubs.pkl'


def test_read_video():
    frames = read_video(VIDEO_PATH)
    print(f"✅ Frames loaded: {len(frames)}")
    print(f"✅ Frame size: {frames[0].shape}")


def test_tracker():
    frames = read_video(VIDEO_PATH)
    tracker = Tracker(MODEL_PATH)
    tracks = tracker.get_object_tracks(
        frames,
        read_from_stub=True,
        stub_path=STUB_PATH
    )
    print(f"✅ Players tracked in frame 0: {len(tracks['players'][0])}")
    print(f"✅ Ball detected in frame 0:   {len(tracks['ball'][0])}")
    print(f"✅ Referees in frame 0:        {len(tracks['referees'][0])}")


def test_single_frame():
    frames = read_video(VIDEO_PATH)
    frame = frames[0]
    cv2.imwrite('output_videos/frame0.jpg', frame)
    print("✅ Frame 0 saved as frame0.jpg — open it to verify video loaded correctly")


if __name__ == '__main__':
    print("── Testing video read ──")
    test_read_video()

    print("\n── Testing tracker ──")
    test_tracker()

    print("\n── Saving single frame ──")
    test_single_frame()