"""
Generates synthetic head-pose data and runs 4 experiments from the proposal:
    1. Threshold Optimization (yaw thresholds × temporal windows)
    2. Lighting Robustness (simulated brightness conditions)
    3. Real-Time Performance (latency measurement)
    4. User Study Simulation (multiple work sessions)

Also generates all figures for the final report.

Usage:
    python run_experiments.py          # Run everything
    python run_experiments.py --exp 1  # Run just experiment 1
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

import cv2
import mediapipe as mp

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


# Synthetic Data Generation

def generate_synthetic_data(fps=30.0):
    """
    Create ~60s of synthetic head-angle data simulating focused work
    with occasional distractions. Each row is one "frame" with a yaw,
    pitch, roll angle and a ground-truth label we defined ourselves.
    """
    np.random.seed(42)
    segments = [
        # (num_frames, label, yaw_mean, yaw_std, pitch_mean, pitch_std)
        (450, "focused",     0,   5,  -5, 3),    # 15s focused work
        (60,  "focused",   -15,   3,   0, 2),    # 2s brief glance left
        (150, "distracted", -50,  8,   5, 5),    # 5s looking away left
        (300, "focused",     0,   6,  -2, 3),    # 10s focused
        (210, "distracted",  55, 10,  10, 5),    # 7s phone distraction right
        (300, "focused",     0,   4,  -3, 3),    # 10s focused
        (120, "distracted", -60,  8,  15, 5),    # 4s talking to someone
        (210, "focused",     1,   5,  -5, 4),    # 7s focused
    ]

    rows = []
    for n_frames, label, ym, ys, pm, ps in segments:
        for _ in range(n_frames):
            rows.append({
                'frame': len(rows),
                'timestamp': len(rows) / fps,
                'yaw': ym + np.random.normal(0, ys),
                'pitch': pm + np.random.normal(0, ps),
                'roll': np.random.normal(0, 3),
                'label': label,
            })

    df = pd.DataFrame(rows)
    # Smooth so angles transition gradually (like real head movement)
    for col in ['yaw', 'pitch', 'roll']:
        df[col] = df[col].rolling(5, center=True, min_periods=1).mean().round(2)
    return df


# Shared: run classifier on angle data

def classify_data(df, yaw_thresh, pitch_thresh, window_sec, fps=30.0):
    """Run threshold + sliding window classifier on a DataFrame of angles."""
    window = int(window_sec * fps)
    history = []
    preds = []

    for _, row in df.iterrows():
        is_dist = abs(row['yaw']) > yaw_thresh or abs(row['pitch']) > pitch_thresh
        history.append(is_dist)
        if len(history) > window:
            history.pop(0)
        ratio = sum(history) / len(history)
        preds.append("distracted" if ratio >= 0.6 else "focused")

    return preds


# Experiment 1: Threshold Optimization 

def exp1_threshold_optimization(df):
    print("\n=== Experiment 1: Threshold Optimization ===")

    yaw_thresholds = [15, 20, 25, 30, 35, 40, 45, 50]
    windows = [1.0, 2.0, 3.0, 5.0]
    results = []

    for yt in yaw_thresholds:
        for w in windows:
            preds = classify_data(df, yt, 25.0, w)
            true = (df['label'] == 'distracted').astype(int).values
            pred = np.array([1 if p == 'distracted' else 0 for p in preds])

            p = precision_score(true, pred, zero_division=0)
            r = recall_score(true, pred, zero_division=0)
            f = f1_score(true, pred, zero_division=0)
            a = accuracy_score(true, pred)
            results.append({'yaw_thresh': yt, 'window': w,
                            'precision': p, 'recall': r, 'f1': f, 'accuracy': a})
            print(f"  Yaw={yt:2d}deg W={w:.0f}s -> P={p:.3f} R={r:.3f} F1={f:.3f} Acc={a:.3f}")

    rdf = pd.DataFrame(results)
    rdf.to_csv(f"{RESULTS}/exp1_results.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for w in windows:
        sub = rdf[rdf['window'] == w]
        axes[0].plot(sub['yaw_thresh'], sub['f1'], 'o-', label=f"W={w:.0f}s")
    axes[0].set(xlabel="Yaw Threshold (°)", ylabel="F1 Score",
                title="F1 vs Yaw Threshold", ylim=(0, 1))
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Confusion matrix for best config
    best = rdf.loc[rdf['f1'].idxmax()]
    preds = classify_data(df, best['yaw_thresh'], 25.0, best['window'])
    true = (df['label'] == 'distracted').astype(int).values
    pred = np.array([1 if p == 'distracted' else 0 for p in preds])
    cm = confusion_matrix(true, pred)

    im = axes[1].imshow(cm, cmap='Blues')
    axes[1].set(xticks=[0, 1], yticks=[0, 1],
                xticklabels=['Focused', 'Distracted'],
                yticklabels=['Focused', 'Distracted'],
                xlabel='Predicted', ylabel='Actual',
                title=f"Best: Yaw={best['yaw_thresh']:.0f}° W={best['window']:.0f}s F1={best['f1']:.3f}")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14,
                         color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/exp1_threshold_optimization.png", dpi=150)
    plt.close()
    print(f"  Best: Yaw={best['yaw_thresh']:.0f}deg W={best['window']:.0f}s -> F1={best['f1']:.3f}")


# Experiment 2: Lighting Robustness 

def exp2_lighting_robustness():
    print("\n=== Experiment 2: Lighting Robustness ===")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, min_detection_confidence=0.5,
        min_tracking_confidence=0.5, refine_landmarks=True,
    )

    # Create a simple synthetic face image
    h, w = 480, 640
    base = np.ones((h, w, 3), dtype=np.uint8) * 180
    cx, cy, r = w // 2, h // 2, 100
    cv2.ellipse(base, (cx, cy), (r, int(r * 1.3)), 0, 0, 360, (210, 180, 160), -1)
    cv2.circle(base, (cx - 35, cy - 25), 8, (60, 40, 30), -1)
    cv2.circle(base, (cx + 35, cy - 25), 8, (60, 40, 30), -1)
    cv2.line(base, (cx, cy - 10), (cx, cy + 15), (150, 120, 100), 2)
    cv2.ellipse(base, (cx, cy + 35), (25, 10), 0, 0, 180, (120, 80, 80), 2)

    conditions = [
        ("Very Dark",   0.2),
        ("Dim",         0.5),
        ("Normal",      1.0),
        ("Bright",      1.5),
        ("Overexposed", 2.5),
    ]

    results = []
    for name, brightness in conditions:
        detections = 0
        for _ in range(30):
            img = np.clip(base * brightness + np.random.normal(0, 5, base.shape), 0, 255).astype(np.uint8)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if res.multi_face_landmarks:
                detections += 1

        rate = detections / 30
        results.append({'condition': name, 'brightness': brightness, 'detection_rate': rate})
        print(f"  {name:14s} (x{brightness:.1f}) -> Detection: {rate:.0%}")

    rdf = pd.DataFrame(results)
    rdf.to_csv(f"{RESULTS}/exp2_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#2ecc71' if r > 0.8 else '#f39c12' if r > 0.5 else '#e74c3c'
              for r in rdf['detection_rate']]
    ax.bar(range(len(rdf)), rdf['detection_rate'], color=colors, tick_label=rdf['condition'])
    ax.set(ylabel="Detection Rate", title="Face Detection Rate vs Lighting", ylim=(0, 1.15))
    ax.axhline(0.9, color='gray', linestyle='--', alpha=0.5, label='90% target')
    ax.legend()
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/exp2_lighting_robustness.png", dpi=150)
    plt.close()
    face_mesh.close()


# Experiment 3: Real-Time Performance

def exp3_performance():
    print("\n=== Experiment 3: Real-Time Performance ===")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, min_detection_confidence=0.5,
        min_tracking_confidence=0.5, refine_landmarks=True,
    )

    # Measure latency on 1280x720 frames (720p webcam resolution)
    h, w = 720, 1280
    n_frames = 50
    latencies = []

    for _ in range(n_frames):
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t0 = time.time()
        face_mesh.process(rgb)
        latencies.append((time.time() - t0) * 1000)

    avg = np.mean(latencies)
    std = np.std(latencies)
    fps = 1000.0 / avg if avg > 0 else 0

    print(f"  Resolution: {w}x{h}")
    print(f"  Avg latency: {avg:.1f}ms ± {std:.1f}ms")
    print(f"  FPS: {fps:.0f}")
    print(f"  Real-time (<100ms): {'YES' if avg < 100 else 'NO'}")

    results = pd.DataFrame([{
        'resolution': f'{w}x{h}', 'avg_ms': round(avg, 1),
        'std_ms': round(std, 1), 'fps': round(fps, 1),
    }])
    results.to_csv(f"{RESULTS}/exp3_results.csv", index=False)

    # Latency distribution plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(latencies, bins=20, color='#3498db', edgecolor='white', alpha=0.8)
    ax.axvline(avg, color='red', linestyle='-', lw=2, label=f'Mean: {avg:.1f}ms')
    ax.axvline(100, color='orange', linestyle='--', lw=1.5, label='100ms threshold')
    ax.set(xlabel="Latency (ms)", ylabel="Count",
           title=f"Processing Latency Distribution ({w}x{h}, {n_frames} frames)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/exp3_performance.png", dpi=150)
    plt.close()
    face_mesh.close()


# Experiment 4: User Study Simulation

def exp4_user_study():
    print("\n=== Experiment 4: User Study Simulation ===")
    np.random.seed(123)
    fps = 30.0

    sessions = [
        ("Deep Focus",      0.05, 25),
        ("Moderate Focus",  0.15, 30),
        ("Frequent Breaks", 0.25, 20),
        ("Tired/Late",      0.35, 25),
        ("Meeting Day",     0.20, 15),
    ]

    results = []
    for name, dist_prob, dur_min in sessions:
        n = int(dur_min * 60 * fps)
        window = int(3.0 * fps)
        history = []
        state = "focused"
        state_frames = 0
        tp = fp = tn = fn = 0

        for i in range(n):
            # Simple state machine: randomly switch between focused/distracted
            state_frames += 1
            if state == "focused" and np.random.random() < dist_prob / fps:
                state = "distracted"
                state_frames = 0
            elif state == "distracted" and state_frames > fps * np.random.uniform(2, 8):
                state = "focused"
                state_frames = 0

            # Generate angles based on current state
            if state == "focused":
                yaw = np.random.normal(0, 8)
            else:
                yaw = np.random.normal(np.random.choice([-45, 45]), 10)

            # Run classifier
            is_dist = abs(yaw) > 30
            history.append(is_dist)
            if len(history) > window:
                history.pop(0)
            pred = "distracted" if sum(history) / len(history) >= 0.6 else "focused"

            # Score
            if state == "distracted" and pred == "distracted": tp += 1
            elif state == "focused" and pred == "distracted":  fp += 1
            elif state == "focused" and pred == "focused":     tn += 1
            else:                                              fn += 1

        total = tp + fp + tn + fn
        p = tp / (tp + fp) if (tp + fp) else 0
        r = tp / (tp + fn) if (tp + fn) else 0
        f = 2 * p * r / (p + r) if (p + r) else 0
        a = (tp + tn) / total

        results.append({'session': name, 'precision': round(p, 3),
                        'recall': round(r, 3), 'f1': round(f, 3), 'accuracy': round(a, 3)})
        print(f"  {name:18s} -> P={p:.3f} R={r:.3f} F1={f:.3f} Acc={a:.3f}")

    rdf = pd.DataFrame(results)
    rdf.to_csv(f"{RESULTS}/exp4_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(rdf))
    w = 0.22
    ax.bar(x - w, rdf['precision'], w, label='Precision', color='#3498db')
    ax.bar(x,     rdf['recall'],    w, label='Recall',    color='#2ecc71')
    ax.bar(x + w, rdf['f1'],       w, label='F1 Score',  color='#e74c3c')
    ax.set(xticks=x, ylabel="Score", title="Classification Metrics Across Sessions", ylim=(0, 1.1))
    ax.set_xticklabels(rdf['session'], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/exp4_user_study.png", dpi=150)
    plt.close()


# Report Figures 

def generate_report_figures(df):
    """Generate architecture diagram, angle distributions, and timeline."""
    print("\n=== Generating Report Figures ===")

    # 1. System Architecture
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis('off')
    boxes = [
        (0.3,  "Webcam\nInput",       "#3498db"),
        (2.7,  "MediaPipe\nFace Mesh", "#2ecc71"),
        (5.1,  "PnP\nPose Est.",      "#f39c12"),
        (7.5,  "Threshold\nClassifier","#e74c3c"),
        (9.9,  "Visual\nAlert",       "#9b59b6"),
    ]
    for x, text, color in boxes:
        from matplotlib.patches import FancyBboxPatch
        ax.add_patch(FancyBboxPatch((x, 0.7), 1.8, 1.4, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='black', alpha=0.85))
        ax.text(x + 0.9, 1.4, text, ha='center', va='center', fontsize=9,
                fontweight='bold', color='white')
    for i in range(len(boxes) - 1):
        ax.annotate("", xy=(boxes[i + 1][0], 1.4), xytext=(boxes[i][0] + 1.8, 1.4),
                     arrowprops=dict(arrowstyle="->", lw=2, color='#333'))
    ax.set_title("System Architecture", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/fig_architecture.png", dpi=150)
    plt.close()

    # 2. Angle Distributions
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    focused = df[df['label'] == 'focused']
    distracted = df[df['label'] == 'distracted']

    for ax, col, thresh, title in [
        (axes[0], 'yaw',   30, 'Yaw (Left-Right)'),
        (axes[1], 'pitch', 25, 'Pitch (Up-Down)'),
    ]:
        ax.hist(focused[col], bins=40, alpha=0.6, label='Focused', color='#2ecc71', density=True)
        ax.hist(distracted[col], bins=40, alpha=0.6, label='Distracted', color='#e74c3c', density=True)
        ax.axvline(thresh, color='k', ls='--', alpha=0.6)
        ax.axvline(-thresh, color='k', ls='--', alpha=0.6, label=f'±{thresh}° thresh')
        ax.set(xlabel=f"{col.title()} (°)", ylabel="Density", title=title)
        ax.legend(fontsize=8)

    axes[2].scatter(focused['yaw'], focused['pitch'], s=2, alpha=0.3, c='#2ecc71', label='Focused')
    axes[2].scatter(distracted['yaw'], distracted['pitch'], s=2, alpha=0.3, c='#e74c3c', label='Distracted')
    axes[2].add_patch(plt.Rectangle((-30, -25), 60, 50, fill=False, ec='black', ls='--', lw=1.5))
    axes[2].set(xlabel="Yaw (°)", ylabel="Pitch (°)", title="Yaw vs Pitch (Focus Zone)")
    axes[2].legend(fontsize=8, markerscale=5)
    plt.tight_layout()
    plt.savefig(f"{RESULTS}/fig_angle_distributions.png", dpi=150)
    plt.close()

    # 3. Timeline (full 60s)
    fig, axes = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    t = df['timestamp']

    axes[0].plot(t, df['yaw'], color='#3498db', lw=0.8)
    axes[0].axhline(30, color='r', ls='--', alpha=0.4); axes[0].axhline(-30, color='r', ls='--', alpha=0.4)
    axes[0].fill_between(t, -30, 30, alpha=0.08, color='green')
    axes[0].set(ylabel="Yaw (°)", title="Head Angles Over Time")
    axes[0].grid(alpha=0.2)

    axes[1].plot(t, df['pitch'], color='#f39c12', lw=0.8)
    axes[1].axhline(25, color='r', ls='--', alpha=0.4); axes[1].axhline(-25, color='r', ls='--', alpha=0.4)
    axes[1].fill_between(t, -25, 25, alpha=0.08, color='green')
    axes[1].set(ylabel="Pitch (°)")
    axes[1].grid(alpha=0.2)

    label_num = (df['label'] == 'distracted').astype(float)
    axes[2].fill_between(t, 0, label_num, alpha=0.5, color='#e74c3c', step='mid', label='Distracted')
    axes[2].fill_between(t, 0, 1 - label_num, alpha=0.3, color='#2ecc71', step='mid', label='Focused')
    axes[2].set(xlabel="Time (s)", ylabel="State", yticks=[0, 1], yticklabels=['Focused', 'Distracted'])
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(f"{RESULTS}/fig_timeline.png", dpi=150)
    plt.close()

    print("  Saved: fig_architecture.png, fig_angle_distributions.png, fig_timeline.png")


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=int, choices=[1, 2, 3, 4], help="Run specific experiment")
    args = parser.parse_args()

    df = generate_synthetic_data()
    print(f"Synthetic data: {len(df)} frames, {df.timestamp.max():.0f}s")

    if args.exp == 1 or args.exp is None:
        exp1_threshold_optimization(df)
    if args.exp == 2 or args.exp is None:
        exp2_lighting_robustness()
    if args.exp == 3 or args.exp is None:
        exp3_performance()
    if args.exp == 4 or args.exp is None:
        exp4_user_study()

    if args.exp is None:
        generate_report_figures(df)
        print(f"\nAll done! Results in {RESULTS}/")


if __name__ == "__main__":
    main()
