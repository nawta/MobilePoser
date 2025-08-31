#!/usr/bin/env python3
"""Test streaming performance with different buffer sizes."""

import os
import time
import subprocess
import sys
from pathlib import Path

# Test configurations
configs = [
    {"buffer": 50, "batch": 512, "accumulate": 16},
    {"buffer": 500, "batch": 1024, "accumulate": 8},
    {"buffer": 2000, "batch": 1024, "accumulate": 8},
    {"buffer": 5000, "batch": 1024, "accumulate": 8},
    {"buffer": 10000, "batch": 2048, "accumulate": 4},
]

def monitor_training(buffer_size, batch_size, accumulate):
    """Run training and monitor performance."""
    print(f"\n{'='*60}")
    print(f"Testing: buffer={buffer_size}, batch={batch_size}, accumulate={accumulate}")
    print(f"{'='*60}")
    
    # Update config
    config_path = Path("mobileposer/config.py")
    config_content = config_path.read_text()
    
    # Replace stream_buffer_size
    import re
    config_content = re.sub(
        r'stream_buffer_size = \d+',
        f'stream_buffer_size = {buffer_size}',
        config_content
    )
    config_content = re.sub(
        r'accumulate_grad_batches = \d+',
        f'accumulate_grad_batches = {accumulate}',
        config_content
    )
    config_path.write_text(config_content)
    
    # Update batch size logic in data.py
    data_path = Path("mobileposer/data.py")
    data_content = data_path.read_text()
    
    # Calculate reduction factor
    reduction = 4096 // batch_size
    data_content = re.sub(
        r'max\(256, original_bs // \d+\)',
        f'max({batch_size//2}, original_bs // {reduction})',
        data_content
    )
    data_path.write_text(data_content)
    
    # Run training and capture output
    cmd = f"PYTHONPATH=. python mobileposer/train.py --module poser --max-seq 500 --stream"
    
    start_time = time.time()
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    max_speed = 0
    speeds = []
    memory_usage = []
    
    # Monitor for 30 seconds
    deadline = time.time() + 30
    
    while time.time() < deadline:
        line = process.stdout.readline()
        if not line:
            break
            
        # Look for speed metrics
        if "it/s" in line:
            try:
                # Extract speed (e.g., "4.77it/s")
                speed_match = re.search(r'(\d+\.\d+)it/s', line)
                if speed_match:
                    speed = float(speed_match.group(1))
                    speeds.append(speed)
                    max_speed = max(max_speed, speed)
            except:
                pass
        
        # Check memory periodically
        if len(speeds) % 10 == 0:
            mem_result = subprocess.run(['free', '-m'], capture_output=True, text=True)
            mem_lines = mem_result.stdout.strip().split('\n')
            mem_used = int(mem_lines[1].split()[2])
            memory_usage.append(mem_used)
    
    process.terminate()
    process.wait()
    
    # Final memory check
    mem_result = subprocess.run(['free', '-m'], capture_output=True, text=True)
    mem_lines = mem_result.stdout.strip().split('\n')
    final_mem = int(mem_lines[1].split()[2])
    
    # Results
    avg_speed = sum(speeds) / len(speeds) if speeds else 0
    max_mem = max(memory_usage) if memory_usage else final_mem
    
    print(f"\nResults:")
    print(f"  Average speed: {avg_speed:.2f} it/s")
    print(f"  Max speed: {max_speed:.2f} it/s")
    print(f"  Max memory: {max_mem/1024:.1f} GB")
    print(f"  Effective batch size: {batch_size * accumulate}")
    
    return {
        "buffer": buffer_size,
        "batch": batch_size,
        "accumulate": accumulate,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "max_memory_gb": max_mem/1024
    }

if __name__ == "__main__":
    results = []
    
    print("Starting streaming performance tests...")
    print(f"System has 125GB RAM available")
    
    for config in configs:
        result = monitor_training(config["buffer"], config["batch"], config["accumulate"])
        results.append(result)
        time.sleep(2)  # Cool down between tests
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'Buffer':>8} | {'Batch':>6} | {'Accum':>6} | {'Eff.BS':>8} | {'Avg Speed':>10} | {'Max Mem':>8}")
    print(f"{'-'*8}-+-{'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}")
    
    for r in results:
        eff_bs = r['batch'] * r['accumulate']
        print(f"{r['buffer']:>8} | {r['batch']:>6} | {r['accumulate']:>6} | {eff_bs:>8} | "
              f"{r['avg_speed']:>8.2f}/s | {r['max_memory_gb']:>6.1f}GB")
    
    # Find optimal
    best = max(results, key=lambda x: x['avg_speed'])
    print(f"\nOptimal configuration:")
    print(f"  Buffer size: {best['buffer']}")
    print(f"  Batch size: {best['batch']}")
    print(f"  Accumulation: {best['accumulate']}")
    print(f"  Speed: {best['avg_speed']:.2f} it/s")
    print(f"  Memory: {best['max_memory_gb']:.1f} GB")