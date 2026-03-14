# Real-Time Attention Monitoring System Using Head Pose Estimation

## Files
| File | Description |
|------|-------------|
| `main.py` | Real-time attention monitor (webcam or video input) |
| `run_experiments.py` | All 4 experiments + report figure generation |
| `requirements.txt` | Python dependencies |

## Setup
```bash
pip install -r requirements.txt
```

## Usage

### Run the monitor (requires webcam or video file)
```bash
python main.py                     # Webcam
python main.py --video test.mp4    # Video file
python main.py --debug             # Show landmarks
python main.py --yaw 35 --pitch 30 # Custom thresholds
```
Controls: `q` quit, `d` toggle debug overlay

### Run experiments & generate figures
```bash
python run_experiments.py          # All experiments + figures
python run_experiments.py --exp 1  # Just threshold optimization
```

## How It Works
1. **MediaPipe Face Mesh** detects 468 facial landmarks in real-time
2. **solvePnP** computes head orientation (yaw/pitch/roll) from 6 key landmarks
3. **Threshold classifier** with sliding window labels frames as focused/distracted
4. **Visual alert** triggers after sustained distraction (flashing border + "REFOCUS!")

## Experiments
1. **Threshold Optimization** — sweep yaw thresholds × temporal windows, measure F1
2. **Lighting Robustness** — test face detection under varied brightness
3. **Real-Time Performance** — measure latency/FPS at different resolutions
4. **User Study Simulation** — simulate 5 work sessions with different distraction levels
