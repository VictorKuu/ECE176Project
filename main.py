"""
This script captures webcam input, detects facial landmarks using MediaPipe,
estimates head pose via PnP, and classifies attention as focused/distracted.
When the user looks away for too long, a visual alert is triggered.

Usage:
    python main.py                  # Run with webcam
    python main.py --video test.mp4 # Run on a video file
    python main.py --debug          # Show landmark overlay

Controls:
    q - Quit
    d - Toggle debug overlay
"""

import argparse
import time
import sys
import os
import cv2
import numpy as np
import mediapipe as mp


# Head Pose Estimation:

# MediaPipe records 468 points on the face. We only take 6:
# nose tip, chin, left eye outer, right eye outer, left mouth, right mouth
KEY_LANDMARKS = [1, 152, 33, 263, 61, 291]

# Canonical 3D face model points (generic face geometry)
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],            # Nose tip
    [0.0, -330.0, -65.0],       # Chin
    [-225.0, 170.0, -135.0],    # Left eye outer
    [225.0, 170.0, -135.0],     # Right eye outer
    [-150.0, -150.0, -125.0],   # Left mouth corner
    [150.0, -150.0, -125.0],    # Right mouth corner
], dtype=np.float64)


def get_head_pose(landmarks_2d, img_w, img_h):
    """
    Compute head yaw/pitch/roll from 6 facial landmarks using solvePnP.
    yaw: face turn left/right
    pitch: face turn up/down
    roll: head tilt

    Args:
        landmarks_2d: (6, 2) array of the 6 key landmark pixel coordinates
        img_w, img_h: frame dimensions

    Returns:
        (yaw, pitch, roll) in degrees, or None if PnP fails
    """
    # Approximate camera intrinsics (pinhole model)
    focal_length = img_w
    cam_matrix = np.array([
        [focal_length, 0, img_w / 2],
        [0, focal_length, img_h / 2],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        FACE_3D_MODEL, landmarks_2d, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    # Rotation vector → rotation matrix → Euler angles
    rmat, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rmat[0, 0]**2 + rmat[1, 0]**2)
    if sy > 1e-6:
        roll = np.degrees(np.arctan2(rmat[2, 1], rmat[2, 2]))
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw = np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0]))
    else:
        roll = np.degrees(np.arctan2(-rmat[1, 2], rmat[1, 1]))
        pitch = np.degrees(np.arctan2(-rmat[2, 0], sy))
        yaw = 0.0

    return yaw, pitch, roll


# Attention Classification:

class AttentionClassifier:
    """
    In the last 3 second frame, if yaw/pitch exceed 30/25 degrees, subject distracted

    Per-frame: distracted if |yaw| > threshold or |pitch| > threshold.
    Temporal: smooths over a window to avoid false triggers on brief glances.
    """

    def __init__(self, yaw_thresh=30.0, pitch_thresh=25.0,
                 window_sec=3.0, trigger_ratio=0.6, fps=30.0):
        self.yaw_thresh = yaw_thresh
        self.pitch_thresh = pitch_thresh
        self.trigger_ratio = trigger_ratio

        self.window_size = int(window_sec * fps)
        self.history = []
        self.distraction_start = None

    def update(self, yaw, pitch, timestamp):
        """
        Classify one frame. Returns (smoothed_state, distraction_seconds).
        smoothed_state: "focused", "distracted", or "no_face"
        """
        if yaw is None:
            is_distracted = True 
        else:
            is_distracted = abs(yaw) > self.yaw_thresh or abs(pitch) > self.pitch_thresh

        # Sliding window
        self.history.append(is_distracted)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # Smoothed decision
        ratio = sum(self.history) / len(self.history)
        smoothed = "distracted" if ratio >= self.trigger_ratio else "focused"

        # Track sustained distraction duration
        if smoothed == "distracted":
            if self.distraction_start is None:
                self.distraction_start = timestamp
            duration = timestamp - self.distraction_start
        else:
            self.distraction_start = None
            duration = 0.0

        return smoothed, duration

    def reset(self):
        self.history.clear()
        self.distraction_start = None


# Visual Feedback (Stimulus)

def draw_alert(frame, duration):
    """Draw flashing border + REFOCUS text when distracted."""
    h, w = frame.shape[:2]

    # Flashing red/orange border
    tick = int(time.time() * 5) % 2
    color = (0, 0, 255) if tick == 0 else (0, 165, 255)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 8)

    # Pulsing text
    scale = 1.5 + 0.3 * np.sin(time.time() * 4)
    text = "REFOCUS!"
    sz = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 3)[0]
    tx, ty = (w - sz[0]) // 2, (h + sz[1]) // 2
    cv2.putText(frame, text, (tx + 2, ty + 2), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), 4)
    cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_DUPLEX, scale, color, 3)

    # Distraction timer bar
    bar_fill = min(duration / 10.0, 1.0)
    cv2.rectangle(frame, (20, h - 40), (20 + int(bar_fill * (w - 40)), h - 20), color, -1)
    cv2.rectangle(frame, (20, h - 40), (w - 20, h - 20), (255, 255, 255), 1)


def draw_hud(frame, yaw, pitch, roll, state, fps, duration, yaw_thresh, pitch_thresh):
    """Draw angle readouts and state indicator."""
    h, w = frame.shape[:2]
    colors = {"focused": (0, 255, 0), "distracted": (0, 0, 255), "no_face": (0, 165, 255)}
    c = colors.get(state, (255, 255, 255))
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (270, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    cv2.putText(frame, f"State: {state.upper()}", (20, 35), font, 0.6, c, 2)
    cv2.putText(frame, f"Yaw:   {yaw:+.1f} (thresh {yaw_thresh:.0f})", (20, 60), font, 0.4,
                (0, 0, 255) if abs(yaw) > yaw_thresh else (0, 255, 0), 1)
    cv2.putText(frame, f"Pitch: {pitch:+.1f} (thresh {pitch_thresh:.0f})", (20, 82), font, 0.4,
                (0, 0, 255) if abs(pitch) > pitch_thresh else (0, 255, 0), 1)
    cv2.putText(frame, f"Roll:  {roll:+.1f}", (20, 104), font, 0.4, (200, 200, 200), 1)
    cv2.putText(frame, f"FPS: {fps:.0f}", (20, 126), font, 0.4, (200, 200, 200), 1)

    if duration > 0:
        cv2.putText(frame, f"Distracted: {duration:.1f}s", (w - 200, 35), font, 0.5, (0, 0, 255), 2)


# Main Loop

def main():
    parser = argparse.ArgumentParser(description="Attention Monitor")
    parser.add_argument("--video", type=str, default=None, help="Video file (default: webcam)")
    parser.add_argument("--yaw", type=float, default=30.0, help="Yaw threshold degrees")
    parser.add_argument("--pitch", type=float, default=25.0, help="Pitch threshold degrees")
    parser.add_argument("--window", type=float, default=3.0, help="Temporal window seconds")
    parser.add_argument("--delay", type=float, default=3.0, help="Seconds before alert triggers")
    parser.add_argument("--debug", action="store_true", help="Show facial landmarks")
    args = parser.parse_args()

    # Open video
    cap = cv2.VideoCapture(args.video if args.video else 0)
    if not cap.isOpened():
        print("Error: cannot open video source. Use --video <path> or connect a webcam.")
        sys.exit(1)

    # Init MediaPipe Face Mesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_landmarks=True,
    )

    classifier = AttentionClassifier(
        yaw_thresh=args.yaw, pitch_thresh=args.pitch,
        window_sec=args.window,
    )

    debug = args.debug
    prev_time = time.time()
    fps = 0.0

    print(f"Thresholds: yaw={args.yaw}°, pitch={args.pitch}°, window={args.window}s")
    print("Controls: q=quit, d=toggle debug")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        dt = now - prev_time
        fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0)
        prev_time = now

        img_h, img_w = frame.shape[:2]

        # Step 1: Detect face landmarks
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        yaw, pitch, roll = 0.0, 0.0, 0.0
        face_found = False

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            face_found = True

            # Extract 6 key landmarks
            pts_2d = np.array([
                [face.landmark[i].x * img_w, face.landmark[i].y * img_h]
                for i in KEY_LANDMARKS
            ], dtype=np.float64)

            # Step 2: Estimate head pose
            angles = get_head_pose(pts_2d, img_w, img_h)
            if angles:
                yaw, pitch, roll = angles

            # Debug: draw landmarks
            if debug:
                for i, lm in enumerate(face.landmark):
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    r = 3 if (i in KEY_LANDMARKS) else 1
                    c = (0, 0, 255) if (i in KEY_LANDMARKS) else (0, 255, 0)
                    cv2.circle(frame, (x, y), r, c, -1)

        # Step 3: Classify attention
        if face_found:
            state, duration = classifier.update(yaw, pitch, now)
        else:
            state, duration = classifier.update(None, None, now)
            state = "no_face"

        # Step 4: Visual feedback
        if state == "distracted" and duration >= args.delay:
            draw_alert(frame, duration)

        draw_hud(frame, yaw, pitch, roll, state, fps, duration,
                 args.yaw, args.pitch)

        cv2.imshow("Attention Monitor", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug = not debug

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()


if __name__ == "__main__":
    main()
