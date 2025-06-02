#!/usr/bin/env python3
"""Benchmark streaming performance with memory monitoring."""

import subprocess
import time
import threading
import psutil
import re

class StreamingBenchmark:
    def __init__(self):
        self.speeds = []
        self.memory_usage = []
        self.gpu_usage = []
        self.running = True
        
    def monitor_resources(self):
        """Monitor system resources in background."""
        while self.running:
            # CPU Memory
            mem = psutil.virtual_memory()
            self.memory_usage.append(mem.used / (1024**3))
            
            # GPU
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True
                )
                gpu_data = result.stdout.strip().split('\n')[0].split(', ')
                self.gpu_usage.append({
                    'util': int(gpu_data[0]),
                    'mem': int(gpu_data[1]) / 1024
                })
            except:
                pass
                
            time.sleep(2)
    
    def run_benchmark(self, duration=60):
        """Run training benchmark."""
        print("=== Streaming Performance Benchmark ===")
        print(f"Configuration:")
        print(f"  stream_buffer_size: 20000")
        print(f"  batch_size: 1024") 
        print(f"  num_workers: 24")
        print(f"  Duration: {duration} seconds")
        print()
        
        # Start resource monitor
        monitor_thread = threading.Thread(target=self.monitor_resources)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Run training
        cmd = ['python', 'mobileposer/train.py', '--module', 'poser', '--stream']
        env = {'PYTHONPATH': '.', 'OMP_NUM_THREADS': '4'}
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env, bufsize=1
        )
        
        start_time = time.time()
        last_update = start_time
        
        try:
            while time.time() - start_time < duration:
                line = process.stdout.readline()
                if not line:
                    break
                    
                # Extract speed
                if "it/s" in line and "Epoch" in line:
                    match = re.search(r'(\d+\.\d+)it/s', line)
                    if match:
                        speed = float(match.group(1))
                        self.speeds.append(speed)
                        
                        # Print progress
                        if time.time() - last_update > 5:
                            recent_avg = sum(self.speeds[-10:]) / len(self.speeds[-10:])
                            mem_avg = sum(self.memory_usage[-5:]) / len(self.memory_usage[-5:])
                            
                            print(f"Speed: {speed:.2f} it/s (avg: {recent_avg:.2f}), "
                                  f"Memory: {mem_avg:.1f} GB", end='\r')
                            last_update = time.time()
                            
        finally:
            process.terminate()
            process.wait()
            self.running = False
            
        # Results
        self.print_results()
    
    def print_results(self):
        """Print benchmark results."""
        print("\n\n=== RESULTS ===")
        
        if self.speeds:
            avg_speed = sum(self.speeds) / len(self.speeds)
            max_speed = max(self.speeds)
            min_speed = min(self.speeds)
            
            # Remove outliers
            sorted_speeds = sorted(self.speeds)
            p10 = sorted_speeds[int(len(sorted_speeds) * 0.1)]
            p90 = sorted_speeds[int(len(sorted_speeds) * 0.9)]
            stable_speeds = [s for s in self.speeds if p10 <= s <= p90]
            stable_avg = sum(stable_speeds) / len(stable_speeds) if stable_speeds else 0
            
            print(f"Speed Statistics:")
            print(f"  Average: {avg_speed:.2f} it/s")
            print(f"  Stable Average (P10-P90): {stable_avg:.2f} it/s")
            print(f"  Max: {max_speed:.2f} it/s")
            print(f"  Min: {min_speed:.2f} it/s")
            print(f"  Samples: {len(self.speeds)}")
        
        if self.memory_usage:
            avg_mem = sum(self.memory_usage) / len(self.memory_usage)
            max_mem = max(self.memory_usage)
            print(f"\nMemory Usage:")
            print(f"  Average: {avg_mem:.1f} GB")
            print(f"  Peak: {max_mem:.1f} GB")
            
        if self.gpu_usage:
            avg_gpu_util = sum(g['util'] for g in self.gpu_usage) / len(self.gpu_usage)
            avg_gpu_mem = sum(g['mem'] for g in self.gpu_usage) / len(self.gpu_usage)
            print(f"\nGPU Usage:")
            print(f"  Average Utilization: {avg_gpu_util:.1f}%")
            print(f"  Average Memory: {avg_gpu_mem:.1f} GB")
            
        # Optimization suggestions
        print("\n=== OPTIMIZATION SUGGESTIONS ===")
        
        if avg_mem < 60:
            print(f"- Memory usage is low ({avg_mem:.1f} GB). Increase stream_buffer_size to 30000-40000")
        
        if self.gpu_usage and avg_gpu_util < 80:
            print(f"- GPU utilization is low ({avg_gpu_util:.1f}%). Consider:")
            print("  - Increasing batch_size if GPU memory allows")
            print("  - Reducing num_workers if CPU is bottleneck")
            
        if stable_avg < 5.0:
            print(f"- Training speed is suboptimal ({stable_avg:.2f} it/s). Try:")
            print("  - Enable mixed precision training")
            print("  - Use torch.compile() for model optimization")
            print("  - Check disk I/O bottlenecks")

if __name__ == "__main__":
    benchmark = StreamingBenchmark()
    benchmark.run_benchmark(duration=60)