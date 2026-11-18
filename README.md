# SafeFusion

Multi-modal (audio + video) threat detection prototype that fuses a vision model and an audio model at the decision level, so an alert can still be raised when the camera is blocked, dark, or occluded.

This repository is the reference implementation for the paper **"SafeFusion: System Architecture for Multi-Modal Threat Detection"**, presented at the 7th International Conference on Data Analytics and Management (ICDAM-2026), London Metropolitan University, UK (Springer).

> **Status:** active development. This is a research prototype under continuous iteration (latest: `safefusion_demo_v11.py`), not a production surveillance system. Interfaces, thresholds and results may change between versions.

## Architecture

Two sensor threads run independently and write scores to shared state. A fusion engine on the main thread reads those scores and produces the alert level.

| Component | Input | Model | Output |
|-----------|--------|--------|--------|
| **VPU** (visual) | Webcam via OpenCV | YOLOv8-Nano (`yolov8n.pt`) | `C_weapon`, `C_person` |
| **APU** (audio) | Microphone via `sounddevice` | YAMNet (pretrained, loaded from TensorFlow Hub, used for inference only) | `P_distress`, matched label |
| **Fusion** (main thread) | Shared scores | Rule table below | `SAFE` / `WARNING` / `HIGH` / `CRITICAL` |

**Why decision-level fusion:** each modality produces its own score, and only the scores are combined. If one modality fails (camera blocked, no microphone), the other keeps working, with no retraining and no system restart.

Weapon COCO classes: `knife`, `scissors`, `baseball bat`. Distress audio is matched by keyword against YAMNet's 521 AudioSet labels (`scream`, `shout`, `yell`, `crying`, `sobbing`, `glass`, `shatter`, `alarm`, `gunshot`, `explosion`, `battle cry`).

## Fusion rules

```
if C_weapon > 0.5:
    CRITICAL
elif C_person > 0.5 and P_distress > 0.6:
    HIGH
elif no visual detections and P_distress > 0.8:
    WARNING
else:
    SAFE
```

## Repository layout

| File | Purpose |
|------|---------|
| `safefusion_demo_v11.py` | **Latest** demo (use this one) |
| `old_versions/` | Earlier iterations (`safefusion_demo.py`, `v2` ... `v10`), kept for development history |
| `evaluate.py` | Evaluation script (see Evaluation below) |
| `audio_check.py` | Quick check that the microphone and audio pipeline work |
| `requirements.txt` | Python dependencies |

## Setup

CPU-only. First run downloads YOLO weights and the YAMNet TF-Hub module (internet needed once). Model weights (`*.pt`) are not stored in this repo.

**Windows (PowerShell)**

```powershell
cd safefusion-ai
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python safefusion_demo_v11.py
```

**macOS / Linux**

```bash
cd safefusion-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python safefusion_demo_v11.py
```

If `python` is not on your PATH, try `py -3` on Windows. Use 64-bit Python 3.10-3.12.

### First-run downloads

- **YOLOv8-Nano:** Ultralytics fetches `yolov8n.pt` into the working directory.
- **YAMNet:** TensorFlow Hub fetches `https://tfhub.dev/google/yamnet/1` into the TF Hub cache.

Warm-up can take one to several minutes the first time. Later launches are much faster.

## Configuration

Edit the flags at the top of `safefusion_demo_v11.py`:

```python
CAMERA_INDEX = 0      # try 1, 2, ... if the wrong camera opens
AUDIO_ENABLED = True  # set False for a video-only run (no mic / no TensorFlow)
```

With `AUDIO_ENABLED = False` the audio thread is not started and `P_distress` is grayed out in the overlay.

## Live demo keys

Click the video window first so OpenCV receives keypresses.

| Key | Effect | Typical banner |
|-----|--------|----------------|
| `w` | Force `C_weapon = 0.95` | **CRITICAL** |
| `s` | Force `P_distress = 0.95` | **HIGH** if a person is in view; **WARNING** if the camera is blocked |
| `n` | Force `P_distress = 0.0` | **SAFE** |
| `c` | Clear overrides | Live sensors again |
| `q` | Quit, release camera, stop threads | - |

Sensor threads keep running under overrides; only the fused values are replaced. The overrides exist so that each fusion state can be shown reliably in a live presentation. They are not part of the evaluation.

## Evaluation (in progress)

`evaluate.py` is the evaluation script and is being extended as the system evolves. Quantitative results and the evaluation scenarios are reported in the paper above; numbers for newer versions will be added here once they are re-measured on the same setup.

```bash
python evaluate.py
```

## Roadmap

- Re-run the evaluation on the latest version (v11) and publish the results here.
- Document the hardware and measurement setup for latency and CPU usage.

## Troubleshooting

**Wrong or black camera**
Change `CAMERA_INDEX`. Close Zoom/Teams/other apps that hold the webcam. On Windows, allow camera access for desktop apps in Privacy settings.

**"Waiting for camera..." never clears**
The selected index failed to open. The app keeps retrying instead of crashing. Cycle indices (`0`, `1`, `2`) and restart.

**No microphone / PortAudio error / TensorFlow import error**
The demo prints a message and continues in video-only mode. Set `AUDIO_ENABLED = False` to skip audio entirely. On Linux you may need PortAudio (`sudo apt install libportaudio2`).

**First run is very slow / window appears late**
Normal: PyTorch, YOLO weights, TensorFlow and YAMNet download and compile. Start the app a few minutes before a presentation so models are cached.

**Low frame rate**
The display loop shows the latest frame and is not tied to model speed, so the displayed frame rate on a CPU-only laptop is lower than inference time per frame. You can lower `YOLO_IMGSZ` (320 by default) further if needed.

**Keys do nothing**
The OpenCV window must be focused. Click inside it, then press `w` / `s` / `n` / `c` / `q`.

**Banner stays SAFE during a real scream**
YAMNet on a laptop microphone is not a production-grade detector. Use `s` as the fallback in a live demo.

**`pip install tensorflow` fails on Apple Silicon / old Python**
Use 64-bit Python 3.10-3.12, or set `AUDIO_ENABLED = False` and demo video plus key overrides only.