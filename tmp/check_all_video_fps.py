#!/usr/bin/env python3
"""Check FPS of multiple Nymeria RGB videos."""

import cv2
from pathlib import Path

rgb_root = Path("/mnt/nas2/naoto/nymeria_dataset/data_video_main_rgb")

# Check multiple sequences
sequences = [
    "20230609_s0_angela_harrell_act3_rk26s0",
    "20230526_s0_ava_carter_act5_rk01s0", 
    "20230526_s0_ava_carter_act4_rk02s0",
    "20230526_s0_ava_carter_act3_rk03s0",
    "20230609_s0_angela_harrell_act2_rk27s0"
]

print("Checking FPS of Nymeria RGB videos")
print("=" * 60)

fps_values = []

for seq in sequences:
    video_path = rgb_root / seq / "video_main_rgb.mp4"
    if video_path.exists():
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            
            fps_values.append(fps)
            
            print(f"\nSequence: {seq}")
            print(f"  FPS: {fps}")
            print(f"  Resolution: {width}x{height}")
            print(f"  Frames: {frame_count}")
            print(f"  Duration: {duration:.2f}s")
            
            cap.release()
        else:
            print(f"\nSequence: {seq} - Failed to open video")
    else:
        print(f"\nSequence: {seq} - Video not found")

print("\n" + "=" * 60)
print(f"Summary: All checked videos have FPS = {set(fps_values)}")

if all(fps == 15.0 for fps in fps_values):
    print("\nCONCLUSION: All Nymeria RGB videos are 15 FPS, not 30 FPS!")
    print("This might be due to:")
    print("1. Post-processing downsampling from original 30 FPS")
    print("2. Storage/bandwidth optimization") 
    print("3. Different recording settings than stated in paper")