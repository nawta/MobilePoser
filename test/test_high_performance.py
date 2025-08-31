#!/usr/bin/env python3
"""Test high-performance streaming configuration."""

import torch
import time
import subprocess
import os

print("=== High Performance Streaming Test ===")
print(f"Current settings:")
print(f"  stream_buffer_size = 50000")
print(f"  batch_size = 1024")
print(f"  accumulate_grad_batches = 4")
print(f"  num_workers = 16")
print(f"  pin_memory = True")
print(f"  prefetch_factor = 4")
print(f"  persistent_workers = True")
print()

# Set environment variables for optimal performance
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Run training
cmd = "PYTHONPATH=. python mobileposer/train.py --module poser --stream"

print("Starting training...")
print("Monitoring performance for 60 seconds...")
print()

process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

start_time = time.time()
speeds = []
last_print_time = start_time

while time.time() - start_time < 60:
    line = process.stdout.readline()
    if not line:
        break
    
    # Look for speed indicators
    if "it/s" in line and "Epoch" in line:
        try:
            # Extract speed
            import re
            speed_match = re.search(r'(\d+\.\d+)it/s', line)
            if speed_match:
                speed = float(speed_match.group(1))
                speeds.append(speed)
                
                # Print update every 5 seconds
                if time.time() - last_print_time > 5:
                    avg_speed = sum(speeds[-10:]) / len(speeds[-10:])
                    print(f"Current speed: {speed:.2f} it/s, Average (last 10): {avg_speed:.2f} it/s")
                    last_print_time = time.time()
        except:
            pass

process.terminate()
process.wait()

# Check final memory
mem_result = subprocess.run(['free', '-m'], capture_output=True, text=True)
mem_lines = mem_result.stdout.strip().split('\n')
mem_used = int(mem_lines[1].split()[2]) / 1024

print()
print("=== Results ===")
if speeds:
    print(f"Average speed: {sum(speeds)/len(speeds):.2f} it/s")
    print(f"Max speed: {max(speeds):.2f} it/s")
    print(f"Min speed: {min(speeds):.2f} it/s")
    print(f"Speed samples: {len(speeds)}")
else:
    print("No speed data collected")
    
print(f"Memory used: {mem_used:.1f} GB")
print()

# GPU utilization check
gpu_result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                           capture_output=True, text=True)
for i, line in enumerate(gpu_result.stdout.strip().split('\n')):
    util, mem_used, mem_total = line.split(', ')
    print(f"GPU {i}: {util}% utilization, {int(mem_used)/1024:.1f}/{int(mem_total)/1024:.1f} GB memory")