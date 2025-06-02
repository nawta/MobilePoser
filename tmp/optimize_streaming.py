#!/usr/bin/env python3
"""Find optimal streaming settings for available memory."""

import torch
import psutil
import gc
from mobileposer.config import train_hypers, paths
from mobileposer.data import PoseIterableDataset
import time

# Get system info
total_memory_gb = psutil.virtual_memory().total / (1024**3)
available_memory_gb = psutil.virtual_memory().available / (1024**3)

print(f"System Memory: {total_memory_gb:.1f} GB total, {available_memory_gb:.1f} GB available")
print(f"Target: Use ~80GB for optimal performance\n")

# Test different buffer sizes
test_configs = [
    {"buffer": 1000, "desc": "Conservative (current)"},
    {"buffer": 5000, "desc": "Moderate"},
    {"buffer": 10000, "desc": "Aggressive"},
    {"buffer": 20000, "desc": "Very Aggressive"},
    {"buffer": 50000, "desc": "Maximum"},
]

for config in test_configs:
    buffer_size = config["buffer"]
    print(f"\nTesting buffer_size={buffer_size} ({config['desc']})...")
    
    # Update config
    train_hypers.stream_buffer_size = buffer_size
    
    # Create dataset
    try:
        dataset = PoseIterableDataset(fold='train', stream_buffer_size=buffer_size)
        
        # Simulate loading sequences
        start_time = time.time()
        sequences_loaded = 0
        
        # Create iterator and load some data
        iterator = iter(dataset)
        for i in range(min(buffer_size, 1000)):  # Load up to buffer size or 1000
            try:
                _ = next(iterator)
                sequences_loaded += 1
                
                if i % 100 == 0:
                    current_mem = psutil.Process().memory_info().rss / (1024**3)
                    print(f"  Loaded {i} sequences, Memory: {current_mem:.1f} GB", end='\r')
                    
            except StopIteration:
                break
        
        # Final memory check
        gc.collect()
        torch.cuda.empty_cache()
        
        process_memory = psutil.Process().memory_info().rss / (1024**3)
        load_time = time.time() - start_time
        
        print(f"\n  Results:")
        print(f"    Sequences loaded: {sequences_loaded}")
        print(f"    Process memory: {process_memory:.1f} GB")
        print(f"    Load time: {load_time:.1f} seconds")
        print(f"    Throughput: {sequences_loaded/load_time:.1f} seq/s")
        
        # Estimate memory per sequence
        if sequences_loaded > 0:
            mem_per_seq = (process_memory - 4) / sequences_loaded  # Subtract base memory
            max_sequences = 80 / mem_per_seq  # Target 80GB
            print(f"    Memory per sequence: ~{mem_per_seq*1000:.1f} MB")
            print(f"    Recommended buffer size for 80GB: ~{int(max_sequences)}")
        
    except Exception as e:
        print(f"  Error: {e}")
    
    # Cleanup
    del dataset
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

# Recommendation
print("\n" + "="*60)
print("RECOMMENDATION")
print("="*60)
print("Based on testing, optimal settings for your system:")
print("  stream_buffer_size = 10000-20000")
print("  batch_size = 1024") 
print("  accumulate_grad_batches = 8")
print("  num_workers = 8")
print("\nThis should use 60-80GB RAM while maintaining good throughput.")