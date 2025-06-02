#!/usr/bin/env python3
"""
Wrapper to properly load pyOrbSlam3 with all dependencies.
"""

import os
import sys
import ctypes

# Pre-load required libraries in the correct order
def preload_libraries():
    """Pre-load shared libraries to avoid dependency issues."""
    lib_paths = [
        "/usr/local/lib/libpango_core.so",
        "/usr/local/lib/libpango_windowing.so",
        "/usr/local/lib/libpango_opengl.so", 
        "/usr/local/lib/libpango_vars.so",
        "/usr/local/lib/libpango_image.so",
        "/usr/local/lib/libpango_display.so",
        "/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/lib/libORB_SLAM3.so",
    ]
    
    for lib_path in lib_paths:
        if os.path.exists(lib_path):
            try:
                ctypes.CDLL(lib_path)
                print(f"✓ Loaded: {os.path.basename(lib_path)}")
            except Exception as e:
                print(f"✗ Failed to load {os.path.basename(lib_path)}: {e}")
        else:
            print(f"✗ Not found: {lib_path}")

# Set up environment
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib:/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

# Add pyOrbSlam3 to path
sys.path.insert(0, '/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/build')

print("Loading libraries...")
preload_libraries()

print("\nImporting pyOrbSlam...")
try:
    import pyOrbSlam
    print("✅ SUCCESS: pyOrbSlam imported!")
    
    # Make it available globally
    sys.modules['pyOrbSlam'] = pyOrbSlam
    
except Exception as e:
    print(f"❌ Failed: {e}")
    raise