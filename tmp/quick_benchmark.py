#!/usr/bin/env python3
"""Quick benchmark for streaming performance."""

import subprocess
import time
import re
import statistics

def run_benchmark(buffer_size, batch_size, accumulate, duration=30):
    """Run a quick benchmark with given settings."""
    print(f"\nTesting: buffer={buffer_size}, batch={batch_size}, accumulate={accumulate}")
    print(f"Effective batch size: {batch_size * accumulate}")
    
    # Update config
    subprocess.run([
        'sed', '-i', 
        f's/stream_buffer_size = .*/stream_buffer_size = {buffer_size}/',
        'mobileposer/config.py'
    ])
    
    subprocess.run([
        'sed', '-i',
        f's/accumulate_grad_batches = .*/accumulate_grad_batches = {accumulate}/',
        'mobileposer/config.py'
    ])
    
    # Update batch size
    subprocess.run([
        'sed', '-i',
        f's/max(1024, original_bs/max({batch_size}, original_bs/',
        'mobileposer/data.py'
    ])
    
    # Run training
    cmd = ['python', 'mobileposer/train.py', '--module', 'poser', '--stream']
    env = {'PYTHONPATH': '.'}
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                           text=True, env=env)
    
    start_time = time.time()
    speeds = []
    
    # Collect speed data
    while time.time() - start_time < duration:
        line = proc.stdout.readline()
        if not line:
            break
            
        # Extract speed
        match = re.search(r'(\d+\.\d+)it/s', line)
        if match:
            speed = float(match.group(1))
            speeds.append(speed)
            print(f"  Speed: {speed:.2f} it/s", end='\r')
    
    proc.terminate()
    proc.wait()
    
    # Get memory usage
    mem_output = subprocess.run(['free', '-m'], capture_output=True, text=True)
    mem_used = int(mem_output.stdout.split('\n')[1].split()[2]) / 1024
    
    if speeds:
        avg_speed = statistics.mean(speeds)
        max_speed = max(speeds)
        print(f"\n  Average: {avg_speed:.2f} it/s")
        print(f"  Max: {max_speed:.2f} it/s")
        print(f"  Memory: {mem_used:.1f} GB")
    else:
        print("\n  No speed data collected")
        avg_speed = max_speed = 0
    
    return avg_speed, max_speed, mem_used

# Test configurations
configs = [
    # Current settings
    {"buffer": 5000, "batch": 1024, "accumulate": 8},
    # More aggressive
    {"buffer": 10000, "batch": 1024, "accumulate": 8},
    {"buffer": 20000, "batch": 2048, "accumulate": 4},
]

print("Running streaming benchmarks...")
print("System memory: 125GB available")

results = []
for cfg in configs:
    avg_speed, max_speed, mem = run_benchmark(cfg["buffer"], cfg["batch"], cfg["accumulate"])
    results.append({
        "config": cfg,
        "avg_speed": avg_speed,
        "max_speed": max_speed,
        "memory_gb": mem
    })
    time.sleep(2)

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
for r in results:
    cfg = r["config"]
    print(f"Buffer: {cfg['buffer']}, Batch: {cfg['batch']}, Accum: {cfg['accumulate']}")
    print(f"  Effective batch size: {cfg['batch'] * cfg['accumulate']}")
    print(f"  Speed: {r['avg_speed']:.2f} it/s (max: {r['max_speed']:.2f})")
    print(f"  Memory: {r['memory_gb']:.1f} GB")
    print()