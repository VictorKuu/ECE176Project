import argparse
import time
import sys
import os
 
import cv2
import numpy as np
 
from head_pose_estimator import HeadPoseEstimator
from attention_classifier import AttentionClassifier, AttentionState
from stimulus_player import StimulusPlayer
from utils.visualization import (
    draw_landmarks, draw_pose_axes, draw_info_panel, draw_attention_bar
)
 
 
class AttentionMonitor:
    """Main application: ties together estimation, classification, stimulus, and UI."""
 
    def __init__(self, args):
        self.args = args
        self.debug = args.debug
 
        # Initialize components
        self.estimator = HeadPoseEstimator(
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.classifier = AttentionClassifier(
            yaw_threshold=args.yaw,
            pitch_threshold=args.pitch,
            temporal_window=args.window,
            stimulus_delay=args.delay,
        )
        self.stimulus = StimulusPlayer(
            video_path=args.stimulus,
            overlay_fraction=0.3,
        )
 
        # FPS tracking
        self._frame_times = []
        self._fps = 0.0
 
        # Logging
        self.log_data = []
 
    def run(self):
        """Main loop."""
        # Open video source
        if self.args.video:
            cap = cv2.VideoCapture(self.args.video)
            source_name = os.path.basename(self.args.video)
        else:
            cap = cv2.VideoCapture(0)
            source_name = "Webcam"
 
        if not cap.isOpened():
            print(f"Error: Cannot open video source: {self.args.video or 'webcam'}")
            print("If running without a webcam, use: python main.py --video <path_to_video>")
            sys.exit(1)
 
        print(f"[AttentionMonitor] Source: {source_name}")
        print(f"[AttentionMonitor] Thresholds: yaw={self.args.yaw}°, pitch={self.args.pitch}°")
        print(f"[AttentionMonitor] Window: {self.args.window}s, Delay: {self.args.delay}s")
        print(f"[AttentionMonitor] Controls: q=quit, d=debug, s=screenshot, r=reset, +/-=yaw threshold")
        print()
 
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if self.args.video:
                        break  # End of video file
                    continue
 
                frame_count += 1
                t_start = time.time()
 
                # --- Core pipeline ---
                # 1. Head pose estimation
                pose = self.estimator.estimate(frame)
 
                # 2. Attention classification
                result = self.classifier.classify(pose)
 
                # 3. Stimulus control
                if result.sustained_distraction and not self.stimulus.is_playing:
                    self.stimulus.trigger()
                    print(f"[{frame_count}] STIMULUS TRIGGERED (distracted {result.distraction_duration:.1f}s)")
                elif not result.sustained_distraction and result.smoothed_state == AttentionState.FOCUSED:
                    if self.stimulus.is_playing:
                        self.stimulus.stop()
                        print(f"[{frame_count}] Stimulus stopped - user refocused")
 
                # --- Rendering ---
                display = frame.copy()
 
                # Stimulus overlay (if active)
                display = self.stimulus.render_overlay(display)
 
                # Debug overlay
                if self.debug and pose is not None:
                    draw_landmarks(
                        display, pose.landmarks,
                        key_indices=HeadPoseEstimator.KEY_LANDMARK_IDS,
                    )
                    cam_matrix = self.estimator._get_camera_matrix(
                        frame.shape[1], frame.shape[0]
                    )
                    draw_pose_axes(
                        display, pose.rotation_vector, pose.translation_vector,
                        cam_matrix, np.zeros((4, 1)),
                    )
 
                # Info panel (always shown)
                t_end = time.time()
                self._update_fps(t_end - t_start)
 
                draw_info_panel(
                    display,
                    yaw=result.yaw, pitch=result.pitch, roll=result.roll,
                    state=result.smoothed_state.value,
                    fps=self._fps,
                    distraction_duration=result.distraction_duration,
                    yaw_thresh=self.classifier.yaw_threshold,
                    pitch_thresh=self.classifier.pitch_threshold,
                )
 
                # Attention bar
                focused_ratio = 1.0 - result.confidence if result.frame_state != AttentionState.DISTRACTED else 0.0
                if result.frame_state == AttentionState.FOCUSED:
                    focused_ratio = 1.0
                draw_attention_bar(display, focused_ratio)
 
                # Log
                self.log_data.append({
                    'frame': frame_count,
                    'timestamp': t_start,
                    'yaw': result.yaw,
                    'pitch': result.pitch,
                    'roll': result.roll,
                    'frame_state': result.frame_state.value,
                    'smoothed_state': result.smoothed_state.value,
                    'sustained': result.sustained_distraction,
                    'latency_ms': (t_end - t_start) * 1000,
                })
 
                # Display
                cv2.imshow("Attention Monitor", display)
 
                # --- Key handling ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('d'):
                    self.debug = not self.debug
                    print(f"Debug overlay: {'ON' if self.debug else 'OFF'}")
                elif key == ord('s'):
                    fname = f"screenshot_{frame_count}.png"
                    cv2.imwrite(fname, display)
                    print(f"Screenshot saved: {fname}")
                elif key == ord('r'):
                    self.classifier.reset()
                    self.stimulus.stop()
                    print("Classifier reset")
                elif key == ord('+') or key == ord('='):
                    self.classifier.yaw_threshold += 5
                    print(f"Yaw threshold: {self.classifier.yaw_threshold}°")
                elif key == ord('-'):
                    self.classifier.yaw_threshold = max(5, self.classifier.yaw_threshold - 5)
                    print(f"Yaw threshold: {self.classifier.yaw_threshold}°")
 
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.estimator.release()
            self.stimulus.release()
            self._save_log()
 
    def _update_fps(self, frame_time: float):
        """Track rolling average FPS."""
        self._frame_times.append(frame_time)
        if len(self._frame_times) > 30:
            self._frame_times.pop(0)
        avg = sum(self._frame_times) / len(self._frame_times)
        self._fps = 1.0 / avg if avg > 0 else 0.0
 
    def _save_log(self):
        """Save session log to CSV."""
        if not self.log_data:
            return
        import pandas as pd
        df = pd.DataFrame(self.log_data)
        log_path = "data/session_log.csv"
        os.makedirs("data", exist_ok=True)
        df.to_csv(log_path, index=False)
        print(f"\nSession log saved to {log_path} ({len(df)} frames)")
 
 
def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-Time Attention Monitoring System"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Path to input video (default: webcam)")
    parser.add_argument("--stimulus", type=str, default=None,
                        help="Path to stimulus video (e.g., subway_surfers.mp4)")
    parser.add_argument("--yaw", type=float, default=30.0,
                        help="Yaw threshold in degrees (default: 30)")
    parser.add_argument("--pitch", type=float, default=25.0,
                        help="Pitch threshold in degrees (default: 25)")
    parser.add_argument("--window", type=float, default=3.0,
                        help="Temporal window in seconds (default: 3)")
    parser.add_argument("--delay", type=float, default=3.0,
                        help="Sustained distraction delay before stimulus (default: 3)")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug overlay on startup")
    return parser.parse_args()
 
 
if __name__ == "__main__":
    args = parse_args()
    monitor = AttentionMonitor(args)
    monitor.run()
