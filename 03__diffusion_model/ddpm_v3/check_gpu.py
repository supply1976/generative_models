#!/usr/bin/env python3
"""
check_gpu.py
-----------
Simple script to check GPU availability and configuration for multi-GPU training.

Usage:
    python check_gpu.py
"""

import tensorflow as tf
import os

def check_gpu_setup():
    """Check GPU configuration and availability."""
    print("=" * 60)
    print("GPU Configuration Check")
    print("=" * 60)
    
    # Check TensorFlow version
    print(f"TensorFlow version: {tf.__version__}")
    
    # Check if CUDA is available
    print(f"CUDA available: {tf.test.is_built_with_cuda()}")
    #print(f"GPU support: {tf.test.is_gpu_available()}")
    
    # List physical devices first to check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPU support: {len(gpus) > 0}")  # Modern way to check GPU availability
    cpus = tf.config.list_physical_devices('CPU')
    
    print(f"\nFound {len(cpus)} CPU(s)")
    for i, cpu in enumerate(cpus):
        print(f"  CPU {i}: {cpu.name}")
    
    print(f"\nFound {len(gpus)} GPU(s)")
    if gpus:
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
            
        # Try to enable memory growth
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("✓ Memory growth enabled for all GPUs")
        except RuntimeError as e:
            print(f"✗ Could not enable memory growth: {e}")
            print("  Note: Memory growth must be set before any operations are performed")
    else:
        print("  No GPUs detected")
    
    # Test MirroredStrategy if multiple GPUs available
    if len(gpus) > 1:
        print(f"\n{'='*60}")
        print("Testing MirroredStrategy")
        print("=" * 60)
        
        try:
            strategy = tf.distribute.MirroredStrategy()
            print(f"✓ MirroredStrategy created successfully")
            print(f"  Number of replicas: {strategy.num_replicas_in_sync}")
            
            # Get device names safely for TF 2.12
            device_names = []
            for device in strategy.extended.worker_devices:
                device_names.append(device)
            print(f"  Devices: {device_names}")
            
            # Test simple computation
            with strategy.scope():
                # Create a simple variable
                var = tf.Variable(1.0)
                print("✓ Variable creation in strategy scope successful")
                
                # Test reduction (TF 2.12 compatible)
                @tf.function
                def test_reduce():
                    return strategy.reduce(tf.distribute.ReduceOp.SUM, var, axis=None)
                
                result = test_reduce()
                print(f"✓ Reduction test successful: {result.numpy()}")
                
        except Exception as e:
            print(f"✗ MirroredStrategy test failed: {e}")
            print(f"  This may indicate GPU communication issues or driver problems")
    
    elif len(gpus) == 1:
        print(f"\n{'='*60}")
        print("Single GPU Configuration")
        print("=" * 60)
        print("✓ Single GPU detected - multi-GPU training not available")
        print("  For multi-GPU training, ensure multiple GPUs are installed and visible")
    
    # Check environment variables
    print(f"\n{'='*60}")
    print("Environment Variables")
    print("=" * 60)
    
    cuda_vars = [
        'CUDA_VISIBLE_DEVICES',
        'TF_FORCE_GPU_ALLOW_GROWTH',
        'TF_CPP_MIN_LOG_LEVEL'
    ]
    
    for var in cuda_vars:
        value = os.environ.get(var, "Not set")
        print(f"  {var}: {value}")
    
    # Recommendations
    print(f"\n{'='*60}")
    print("Recommendations")
    print("=" * 60)
    
    if len(gpus) > 1:
        print("✓ Multi-GPU training available!")
        print("  Use: python run.py --config config.yaml --training --multi_gpu")
        print("  Consider adjusting batch size for optimal GPU utilization")
    elif len(gpus) == 1:
        print("✓ Single GPU training available")
        print("  Use: python run.py --config config.yaml --training")
    else:
        print("✗ No GPUs available - training will use CPU")
        print("  Install CUDA and compatible GPU drivers")
        
    print("\nFor optimal performance:")
    print("  - Ensure GPU memory growth is enabled")
    print("  - Consider using --enable_xla for XLA JIT compilation")
    print("  - Monitor GPU utilization during training")

if __name__ == "__main__":
    check_gpu_setup()
