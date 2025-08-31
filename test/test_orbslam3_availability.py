#!/usr/bin/env python3
"""
Test script to check pyOrbSlam3 availability and provide debugging information.
"""

import sys
import os
from pathlib import Path

def test_pyorbslam3():
    """Test if pyOrbSlam3 can be imported and used."""
    
    print("=" * 60)
    print("pyOrbSlam3 Availability Test")
    print("=" * 60)
    
    # Add various potential paths
    potential_paths = [
        "/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/build",
        "./third_party/pyOrbSlam3/pyOrbSlam3/build",
        "third_party/pyOrbSlam3/pyOrbSlam3/build"
    ]
    
    # Find the .so file
    so_file = None
    for base_path in potential_paths:
        if os.path.exists(base_path):
            for file in os.listdir(base_path):
                if file.startswith("pyOrbSlam") and file.endswith(".so"):
                    so_file = os.path.join(base_path, file)
                    break
    
    if so_file and os.path.exists(so_file):
        print(f"✓ Found pyOrbSlam3 module: {so_file}")
        print(f"  Size: {os.path.getsize(so_file)} bytes")
        print(f"  Permissions: {oct(os.stat(so_file).st_mode)[-3:]}")
    else:
        print("✗ pyOrbSlam3 module (.so file) not found")
        return False
    
    # Check vocabulary file
    vocab_path = Path("/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/modules/ORB_SLAM3/Vocabulary/ORBvoc.txt")
    if vocab_path.exists():
        print(f"✓ ORB vocabulary found: {vocab_path}")
        print(f"  Size: {vocab_path.stat().st_size / (1024*1024):.1f} MB")
    else:
        print("✗ ORB vocabulary not found")
    
    # Check config files
    config_base = Path("/home/naoto/docker_workspace/MobilePoser/mobileposer/slam_configs")
    if config_base.exists():
        configs = list(config_base.glob("*.yaml"))
        print(f"✓ Found {len(configs)} SLAM config files:")
        for cfg in configs[:3]:
            print(f"  - {cfg.name}")
    
    # Try importing with different methods
    print("\nAttempting to import pyOrbSlam3...")
    
    # Method 1: Direct path manipulation
    for path in potential_paths:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
    
    # Method 2: Working directory change
    original_cwd = os.getcwd()
    build_dir = "/home/naoto/docker_workspace/MobilePoser/third_party/pyOrbSlam3/pyOrbSlam3/build"
    
    try:
        if os.path.exists(build_dir):
            os.chdir(build_dir)
            
        import pyOrbSlam
        print("✓ Successfully imported pyOrbSlam!")
        
        # Test basic functionality
        print("\nTesting basic functionality:")
        
        # Test Debug class
        db = pyOrbSlam.Debug()
        pid = db.getPID()
        print(f"  Debug.getPID(): {pid}")
        
        print("\n✅ pyOrbSlam3 is available and functional!")
        return True
        
    except ImportError as e:
        print(f"✗ Failed to import pyOrbSlam: {e}")
        
        # Additional debugging
        print("\nDebugging information:")
        print(f"  Python version: {sys.version}")
        print(f"  sys.path includes:")
        for p in sys.path[:5]:
            print(f"    - {p}")
        
        # Check if it's a linking issue
        print("\nChecking for missing dependencies...")
        import subprocess
        if so_file:
            try:
                result = subprocess.run(
                    ["ldd", so_file], 
                    capture_output=True, 
                    text=True
                )
                missing = [line for line in result.stdout.split('\n') if 'not found' in line]
                if missing:
                    print("  Missing libraries:")
                    for lib in missing:
                        print(f"    - {lib}")
                else:
                    print("  All libraries found")
            except:
                print("  Could not check dependencies")
        
        return False
        
    except Exception as e:
        print(f"✗ Unexpected error: {type(e).__name__}: {e}")
        return False
        
    finally:
        os.chdir(original_cwd)


def test_slam_integration():
    """Test SLAM integration with MobilePoser."""
    print("\n" + "=" * 60)
    print("SLAM Integration Test")
    print("=" * 60)
    
    # Import MobilePoser SLAM modules
    try:
        from mobileposer.models.slam import create_slam_interface
        print("✓ Imported SLAM interface")
        
        # Try creating different SLAM types
        print("\nTesting SLAM interface creation:")
        
        # Mock SLAM (should always work)
        mock_slam = create_slam_interface("mock")
        print("✓ Created mock SLAM interface")
        
        # Real SLAM (may fail if pyOrbSlam3 not available)
        try:
            real_slam = create_slam_interface("real")
            print("✓ Created real SLAM interface")
        except Exception as e:
            print(f"ℹ Real SLAM not available: {e}")
            
    except ImportError as e:
        print(f"✗ Failed to import SLAM modules: {e}")


if __name__ == "__main__":
    # Set environment variables
    os.environ["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libstdc++.so.6"
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib/x86_64-linux-gnu:/usr/local/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
    
    # Run tests
    pyorbslam_available = test_pyorbslam3()
    test_slam_integration()
    
    print("\n" + "=" * 60)
    if pyorbslam_available:
        print("Summary: pyOrbSlam3 is available! Real SLAM tests can be run.")
    else:
        print("Summary: pyOrbSlam3 not available. Using mock SLAM for testing.")
        print("\nTo enable real SLAM:")
        print("1. Ensure pyOrbSlam3 is built correctly")
        print("2. Run: source setup_orbslam3_env.sh")
        print("3. Check for missing dependencies with ldd")
    print("=" * 60)