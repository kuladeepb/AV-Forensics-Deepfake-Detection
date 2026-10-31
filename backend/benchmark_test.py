"""
AV-Forensics: Benchmark Evaluation Script
Evaluates the trained multi-modal model across test sets of real and deepfake videos.
Can be executed standalone on any machine with model.pth in place.
"""

import sys
import os
import argparse
from pathlib import Path

# Add project roots to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "ml"))

from inference import analyze_video

def find_test_videos(custom_dir=None):
    candidates = []
    if custom_dir and os.path.exists(custom_dir):
        candidates.append(Path(custom_dir))
    
    env_dir = os.getenv("TEST_VIDEO_DIR")
    if env_dir and os.path.exists(env_dir):
        candidates.append(Path(env_dir))
        
    candidates.extend([
        Path(r"c:\Users\HP\Downloads\deepfakevideos"),
        PROJECT_ROOT / "samples",
        PROJECT_ROOT / "deepfakevideos",
        Path.home() / "Downloads" / "deepfakevideos",
    ])

    test_root = None
    for c in candidates:
        if c.exists() and (c / "real_videos").exists() and (c / "videos").exists():
            test_root = c
            break

    if not test_root:
        return []

    real_dir = test_root / "real_videos"
    fake_dir = test_root / "videos"

    test_list = []
    if real_dir.exists():
        for f in sorted(os.listdir(real_dir)):
            if f.lower().endswith((".mp4", ".mov", ".avi")):
                test_list.append(("REAL", str(real_dir / f)))

    if fake_dir.exists():
        for f in sorted(os.listdir(fake_dir)):
            if f.lower().endswith((".mp4", ".mov", ".avi")):
                test_list.append(("FAKE", str(fake_dir / f)))

    return test_list

def run_benchmark(custom_dir=None):
    videos = find_test_videos(custom_dir)
    if not videos:
        print("[!] No evaluation video directories found.")
        print("    Place your test videos in a folder with subfolders 'real_videos' and 'videos'.")
        print("    Usage: python benchmark_test.py --test-dir <path_to_folder>")
        return

    print("=" * 95)
    print("                    AV-FORENSICS BENCHMARK EVALUATION                     ")
    print("=" * 95)
    print(f"{'FILE':<14} | {'EXPECTED':<8} | {'VERDICT':<8} | {'CONF':<7} | {'MODALITY':<18} | {'AF':<6} | {'VF':<6} | {'STATUS'}")
    print("-" * 95)

    real_total = 0
    real_correct = 0
    fake_total = 0
    fake_correct = 0

    for exp, path in videos:
        name = os.path.basename(path)
        try:
            res = analyze_video(path, name)
            v = res["verdict"]
            conf = f"{res['confidence']*100:.1f}%"
            mod = res["modality"]["label"]
            af = f"{res['modality']['audio_fake_prob']*100:.1f}%"
            vf = f"{res['modality']['video_fake_prob']*100:.1f}%"
            is_correct = (v == exp)
            status = "PASS [OK]" if is_correct else "FAIL [X]"

            if exp == "REAL":
                real_total += 1
                if is_correct:
                    real_correct += 1
            else:
                fake_total += 1
                if is_correct:
                    fake_correct += 1

            print(f"{name:<14} | {exp:<8} | {v:<8} | {conf:<7} | {mod:<18} | {af:<6} | {vf:<6} | {status}", flush=True)
        except Exception as e:
            print(f"{name:<14} | {exp:<8} | ERROR    | {str(e)[:30]:<30}", flush=True)

    print("=" * 95)
    total_samples = real_total + fake_total
    total_correct = real_correct + fake_correct
    acc = (total_correct / total_samples * 100) if total_samples > 0 else 0
    real_acc = (real_correct / real_total * 100) if real_total > 0 else 0
    fake_acc = (fake_correct / fake_total * 100) if fake_total > 0 else 0

    print(f"BENCHMARK SUMMARY:")
    print(f"  * Real Videos Accuracy: {real_correct}/{real_total} ({real_acc:.1f}%)")
    print(f"  * Fake Videos Accuracy: {fake_correct}/{fake_total} ({fake_acc:.1f}%)")
    print(f"  * Overall Accuracy:     {total_correct}/{total_samples} ({acc:.1f}%)")
    print("=" * 95)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AV-Forensics Benchmark Evaluation")
    parser.add_argument("--test-dir", type=str, default=None, help="Directory containing real_videos and videos subfolders")
    args = parser.parse_args()
    run_benchmark(args.test_dir)
