# ⚽ Football Analytics MVP

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![YOLO11](https://img.shields.io/badge/Model-YOLO11_Large-brightgreen?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/Computer_Vision-OpenCV-orange?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/Data-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A computer vision pipeline that turns raw football footage into real-time analytics.**
Player tracking · Team separation · Ball possession · Annotated video output

</div>

---

## 🎬 Demo

| Raw Input               | Annotated Output                        |
| ----------------------- | --------------------------------------- |
| Plain broadcast footage | Player IDs + team colors + possession % |

> Drop your own football clip in `input_videos/` and run `main.py` — that's it.

---

## 🧠 How It Works

```
Raw Video
    │
    ▼
┌─────────────────────┐
│   YOLO11 Detection  │  ← Detects players, referees, ball every frame
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   ByteTrack IDs     │  ← Assigns persistent IDs across frames
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  K-Means Clustering │  ← Reads jersey colors → separates into 2 teams
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Possession Engine  │  ← Closest player to ball = has possession
└─────────────────────┘
    │
    ▼
Annotated Output Video
```

---

## ✨ Features

- 🎯 **Player & Ball Detection** — YOLO11 Large model for high accuracy detection
- 🔁 **Persistent ID Tracking** — ByteTrack keeps player IDs stable across frames
- 👕 **Auto Team Assignment** — K-Means clustering on jersey pixels, no manual labeling
- ⚽ **Ball Possession Logic** — Frame-by-frame proximity-based possession detection
- 🌀 **Occlusion Handling** — Pandas interpolation fills missing ball positions
- 📊 **Live Scoreboard** — Real-time possession % overlay on output video
- 💾 **Stub Caching** — Saves detection results so YOLO doesn't re-run every test

---

## 📂 Project Structure

```
FootBall-Analytics-MVP/
│
├── main.py                        ← Full pipeline entry point
├── proj.py                        ← Dev/test script for individual modules
├── requirements.txt               ← All dependencies
├── yolo11l.pt                     ← YOLO11 Large model weights
│
├── tracker/
│   ├── __init__.py
│   └── tracker.py                 ← YOLO11 inference + ByteTrack ID persistence
│
├── team_assigner/
│   ├── __init__.py
│   └── team_assigner.py           ← K-Means jersey color clustering
│
├── player_ball_assigner/
│   ├── __init__.py
│   └── player_ball_assigner.py    ← Proximity possession + interpolation
│
├── utils/
│   ├── __init__.py
│   ├── video_utils.py             ← read_video(), save_video()
│   └── bbox_utils.py             ← BBox math helpers
│
├── input_videos/                  ← Drop your football clip here
├── output_videos/                 ← Annotated output saved here
└── stubs/                         ← Cached detection results (.pkl)
```

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/FootBall-Analytics-MVP.git
cd FootBall-Analytics-MVP
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create required folders

```bash
mkdir input_videos output_videos stubs
```

### 4. Add your video

Drop a football match clip into `input_videos/` and name it `sample.mp4`.

### 5. Run

```bash
# Quick sanity check first
python proj.py

# Full pipeline
python main.py
```

Output will be saved to `output_videos/output.avi` ✅

---

## ⚙️ Configuration

| Parameter                  | Location                          | Default | Description                    |
| -------------------------- | --------------------------------- | ------- | ------------------------------ |
| `conf`                     | `tracker.py` → `detect_frames()`  | `0.1`   | YOLO confidence threshold      |
| `max_player_ball_distance` | `player_ball_assigner.py`         | `70px`  | Possession proximity threshold |
| `batch_size`               | `tracker.py` → `detect_frames()`  | `20`    | Frames per YOLO batch          |
| `fps`                      | `video_utils.py` → `save_video()` | `24`    | Output video frame rate        |

---

## 📦 Dependencies

| Package         | Version   | Purpose                         |
| --------------- | --------- | ------------------------------- |
| `ultralytics`   | 8.3.0     | YOLO11 detection model          |
| `opencv-python` | 4.10.0.84 | Video I/O + frame annotation    |
| `supervision`   | 0.22.0    | ByteTrack ID persistence        |
| `scikit-learn`  | 1.5.1     | K-Means team clustering         |
| `pandas`        | 2.2.2     | Ball position interpolation     |
| `numpy`         | 1.26.4    | Array math                      |
| `torch`         | 2.8.0     | YOLO11 backend (auto-installed) |

---

## 🔍 Module Deep Dive

### Tracker (`tracker/tracker.py`)

Wraps YOLO11 and supervision's ByteTrack. Runs detection in batches of 20 frames for efficiency. Saves results to a `.pkl` stub so you don't re-run YOLO on every test — huge time saver during development.

### Team Assigner (`team_assigner/team_assigner.py`)

Crops the top half of each player's bounding box (jersey region), runs KMeans(k=2) on the pixels. Uses corner pixels to identify background vs jersey. Runs full team separation on frame 0, then caches each player's team ID for the rest of the video.

### Player Ball Assigner (`player_ball_assigner/player_ball_assigner.py`)

Each frame, measures distance from the ball center to both bottom corners of every player bbox (feet position). The closest player within `max_player_ball_distance` gets possession. Returns `-1` if no player is close enough — last known possession carries forward.

---

## 🛠️ Troubleshooting

**`pip not found`**

```bash
python3 -m pip install -r requirements.txt
```

**`No module named ultralytics`**

```bash
pip3 install ultralytics
```

**Players switching teams mid-video**
→ Increase `n_init` in KMeans inside `team_assigner.py` for more stable clustering.

**Ball possession feels jumpy**
→ Tune `max_player_ball_distance` in `player_ball_assigner.py` (default: 70px).

**Output video won't open**
→ Try changing `XVID` to `mp4v` in `video_utils.py` and saving as `.mp4`.

---

## 🗺️ Roadmap

- [ ] Speed estimation (pixels/frame → km/h)
- [ ] Heatmap generation per player
- [ ] Pass detection and pass network visualization
- [ ] Tactical formation detection
- [ ] Web dashboard for analytics output

---

## 🙌 Acknowledgements

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) — detection backbone
- [Roboflow Supervision](https://github.com/roboflow/supervision) — ByteTrack implementation

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Built with ⚽ + 🐍 + ❤️
</div>
