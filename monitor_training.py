#!/usr/bin/env python3
"""Monitor training with GPU memory tracking."""

import subprocess
import threading
import time
import os

class TrainingMonitor:
    def __init__(self):
        self.running = True
        self.max_gpu_mem = [0, 0]
        
    def monitor_gpu(self):
        """Monitor GPU memory usage."""
        while self.running:
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                    capture_output=True, text=True
                )
                memories = [int(m) for m in result.stdout.strip().split('\n')]
                for i, mem in enumerate(memories[:2]):
                    self.max_gpu_mem[i] = max(self.max_gpu_mem[i], mem)
            except:
                pass
            time.sleep(1)
    
    def run(self):
        """Run training with monitoring."""
        print("Starting training with GPU monitoring...")
        print("Press Ctrl+C to stop\n")
        
        # Start GPU monitor
        monitor_thread = threading.Thread(target=self.monitor_gpu)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Run training
        env = dict(os.environ)
        env['PYTHONPATH'] = '.'
        env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        cmd = ['python', 'mobileposer/train.py', '--module', 'poser', '--stream']
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True, 
            env=env,
            bufsize=1
        )
        
        try:
            start_time = time.time()
            last_speed = 0
            
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                    
                # Check for OOM
                if 'out of memory' in line.lower() or 'oom' in line.lower():
                    print(f"\n!!! OOM DETECTED !!!")
                    print(f"GPU 0 max memory: {self.max_gpu_mem[0]/1024:.1f} GB")
                    print(f"GPU 1 max memory: {self.max_gpu_mem[1]/1024:.1f} GB")
                    print(f"Error line: {line.strip()}")
                    break
                
                # Track speed
                if 'it/s' in line and 'Epoch' in line:
                    try:
                        import re
                        match = re.search(r'(\d+\.\d+)it/s', line)
                        if match:
                            last_speed = float(match.group(1))
                            elapsed = int(time.time() - start_time)
                            print(f"\rSpeed: {last_speed:.2f} it/s, "
                                  f"GPU0: {self.max_gpu_mem[0]/1024:.1f}GB, "
                                  f"GPU1: {self.max_gpu_mem[1]/1024:.1f}GB, "
                                  f"Time: {elapsed}s", end='', flush=True)
                    except:
                        pass
                        
        except KeyboardInterrupt:
            print("\n\nStopped by user")
        finally:
            self.running = False
            process.terminate()
            process.wait()
            
        print(f"\n\nMax GPU Memory Usage:")
        print(f"  GPU 0: {self.max_gpu_mem[0]/1024:.1f} GB / 24.0 GB")
        print(f"  GPU 1: {self.max_gpu_mem[1]/1024:.1f} GB / 24.0 GB")

if __name__ == "__main__":
    monitor = TrainingMonitor()
    monitor.run()