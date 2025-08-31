#!/usr/bin/env python3
"""Run optimized streaming training with monitoring."""

import subprocess
import time
import re
import os

# Set environment for optimal performance
os.environ.update({
    'OMP_NUM_THREADS': '4',
    'MKL_NUM_THREADS': '4', 
    'CUDA_LAUNCH_BLOCKING': '0',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
})

print("=== Optimized Streaming Training ===")
print("Configuration:")
print("  stream_buffer_size: 5000")
print("  batch_size: 1024")
print("  accumulate_grad_batches: 8")
print("  num_workers: 12")
print("  Effective batch size: 8192")
print()

# Run training with real-time monitoring
cmd = ['python', 'mobileposer/train.py', '--module', 'poser', '--stream']

process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env=dict(os.environ, PYTHONPATH='.'),
    bufsize=1
)

speeds = []
start_time = time.time()
last_print = start_time

print("Starting training...\n")

try:
    while True:
        line = process.stdout.readline()
        if not line:
            break
            
        # Print important lines
        if any(keyword in line for keyword in ['Error', 'error', 'WARNING', 'Module Path']):
            print(line.strip())
            
        # Extract and display speed
        if 'it/s' in line and 'Epoch' in line:
            match = re.search(r'(\d+\.\d+)it/s', line)
            if match:
                speed = float(match.group(1))
                speeds.append(speed)
                
                # Update display every 5 seconds
                if time.time() - last_print > 5:
                    recent_speeds = speeds[-20:] if len(speeds) > 20 else speeds
                    avg_speed = sum(recent_speeds) / len(recent_speeds)
                    
                    # Memory check
                    mem_result = subprocess.run(
                        ['free', '-m'], 
                        capture_output=True, 
                        text=True
                    )
                    mem_used = int(mem_result.stdout.split('\n')[1].split()[2]) / 1024
                    
                    print(f"\rSpeed: {speed:.2f} it/s (avg: {avg_speed:.2f}), "
                          f"Memory: {mem_used:.1f} GB, "
                          f"Time: {int(time.time() - start_time)}s", end='', flush=True)
                    
                    last_print = time.time()
                    
except KeyboardInterrupt:
    print("\n\nTraining interrupted by user")
    
process.terminate()
process.wait()

# Final summary
print("\n\n=== Training Summary ===")
if speeds:
    print(f"Average speed: {sum(speeds)/len(speeds):.2f} it/s")
    print(f"Max speed: {max(speeds):.2f} it/s")
    print(f"Total iterations: {len(speeds)}")
    print(f"Training time: {int(time.time() - start_time)} seconds")
else:
    print("No speed data collected")

# GPU summary
gpu_result = subprocess.run(
    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
    capture_output=True, text=True
)
print("\nGPU Status:")
for i, line in enumerate(gpu_result.stdout.strip().split('\n')):
    parts = line.split(', ')
    if len(parts) == 3:
        util, mem_used, mem_total = parts
        print(f"  GPU {i}: {util}% util, {int(mem_used)/1024:.1f}/{int(mem_total)/1024:.1f} GB")