#!/usr/bin/env python3
"""
Fixed pyOrbSlam3 import wrapper that properly handles library loading.
"""

import os
import sys

# Set up environment with proper library paths
def setup_environment():
    """Set up environment for pyOrbSlam3."""
    # Get current LD_LIBRARY_PATH
    current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    
    # Add required library paths
    lib_paths = [
        '/usr/local/lib',
        '/usr/lib/x86_64-linux-gnu',
        '/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/lib',
        '/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Thirdparty/DBoW2/lib',
        '/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Thirdparty/g2o/lib',
    ]
    
    # Combine paths
    new_ld_path = ':'.join(lib_paths) + ':' + current_ld_path
    os.environ['LD_LIBRARY_PATH'] = new_ld_path
    
    # Add pyOrbSlam3 build directory to Python path
    pyorb_build_path = '/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/build'
    if pyorb_build_path not in sys.path:
        sys.path.insert(0, pyorb_build_path)
    
    print("Environment setup complete:")
    print(f"  LD_LIBRARY_PATH: {os.environ['LD_LIBRARY_PATH']}")
    print(f"  Python path includes: {pyorb_build_path}")

# Setup environment before importing
setup_environment()

# Try to import pyOrbSlam
try:
    import pyOrbSlam
    print("\n✅ SUCCESS: pyOrbSlam imported!")
    
    # Make it available globally
    sys.modules['pyOrbSlam'] = pyOrbSlam
    
except Exception as e:
    print(f"\n❌ Failed to import pyOrbSlam: {e}")
    print("\nTrying alternative method...")
    
    # Try with ctypes preloading
    import ctypes
    
    # Pre-load Pangolin libraries in order
    pangolin_libs = [
        'libpango_core.so.0',
        'libpango_opengl.so.0',
        'libpango_windowing.so.0',
        'libpango_image.so.0',
        'libpango_vars.so.0',
        'libpango_display.so.0',
    ]
    
    for lib in pangolin_libs:
        try:
            ctypes.CDLL(f'/usr/local/lib/{lib}', ctypes.RTLD_GLOBAL)
            print(f"  ✓ Loaded {lib}")
        except Exception as e:
            print(f"  ✗ Failed to load {lib}: {e}")
    
    # Try again
    try:
        import pyOrbSlam
        print("\n✅ SUCCESS: pyOrbSlam imported after preloading!")
        sys.modules['pyOrbSlam'] = pyOrbSlam
    except Exception as e:
        print(f"\n❌ Still failed: {e}")
        raise