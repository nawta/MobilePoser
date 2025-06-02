#!/usr/bin/env python3
"""
Test script to import pyOrbSlam3 with proper environment setup.
"""

import os
import sys

# Set up environment
os.environ['LD_LIBRARY_PATH'] = '/usr/local/lib:/usr/lib/x86_64-linux-gnu:' + os.environ.get('LD_LIBRARY_PATH', '')

# Add pyOrbSlam3 build directory to Python path
sys.path.insert(0, '/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/build')

print("Environment setup:")
print(f"  LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")
print(f"  Python path includes: {sys.path[0]}")

try:
    import pyOrbSlam
    print("\n✅ SUCCESS: pyOrbSlam imported successfully!")
    
    # Test basic functionality
    print("\nTesting pyOrbSlam functionality:")
    
    # Test Debug class
    db = pyOrbSlam.Debug()
    pid = db.getPID()
    print(f"  Debug.getPID(): {pid}")
    
    # Check vocabulary path
    vocab_path = "/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Vocabulary/ORBvoc.txt"
    config_path = "/home/naoto/docker_workspace/MobilePoser/mobileposer/slam_configs/nymeria_mono_base.yaml"
    
    if os.path.exists(vocab_path):
        print(f"  ✓ Vocabulary found: {vocab_path}")
    else:
        print(f"  ✗ Vocabulary not found: {vocab_path}")
        
    if os.path.exists(config_path):
        print(f"  ✓ Config found: {config_path}")
    else:
        print(f"  ✗ Config not found: {config_path}")
    
    print("\n✅ pyOrbSlam3 is ready to use!")
    
except ImportError as e:
    print(f"\n❌ Failed to import pyOrbSlam: {e}")
    print("\nTroubleshooting:")
    print("1. Check if LD_LIBRARY_PATH includes /usr/local/lib")
    print("2. Run: ldd third_party/pyOrbSlam3/pyOrbSlam3/build/pyOrbSlam*.so")
    print("3. Look for missing libraries")
except Exception as e:
    print(f"\n❌ Unexpected error: {type(e).__name__}: {e}")