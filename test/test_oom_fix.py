#!/usr/bin/env python3
"""Test OOM fix for streaming training."""

import subprocess
import os

print("=== Testing OOM Fix ===")
print("Adjusted settings:")
print("  stream_buffer_size: 1000 (reduced from 5000)")
print("  batch_size: 512 (reduced from 1024)")
print("  accumulate_grad_batches: 16 (increased from 8)")
print("  num_workers: 8 (reduced from 12)")
print("  prefetch_factor: 2 (reduced from 4)")
print("  Effective batch size: 512 * 16 = 8192")
print()

# Set memory-friendly environment
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

# Test with fast-dev-run first
print("Running quick test...")
result = subprocess.run(
    ['python', 'mobileposer/train.py', '--module', 'poser', '--stream', '--fast-dev-run'],
    env=dict(os.environ, PYTHONPATH='.'),
    capture_output=True,
    text=True
)

if result.returncode != 0:
    print("Error in fast-dev-run:")
    print(result.stderr[-1000:] if result.stderr else result.stdout[-1000:])
else:
    print("Fast-dev-run successful!")
    
# Check GPU memory
gpu_result = subprocess.run(
    ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
    capture_output=True, text=True
)
print("\nGPU Memory Status:")
for i, line in enumerate(gpu_result.stdout.strip().split('\n')):
    used, total = line.split(', ')
    print(f"  GPU {i}: {int(used)/1024:.1f}/{int(total)/1024:.1f} GB ({int(used)*100//int(total)}% used)")