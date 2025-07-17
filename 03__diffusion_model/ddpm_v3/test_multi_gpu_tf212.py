#!/usr/bin/env python3
"""
test_multi_gpu_tf212.py
-----------------------
Test script to verify TensorFlow 2.12 compatibility for multi-GPU training.

Usage:
    python3 test_multi_gpu_tf212.py
"""

import os
import sys
import tensorflow as tf
import numpy as np
from tensorflow import keras

def test_tf_version():
    """Test TensorFlow version compatibility."""
    print("=" * 60)
    print("TensorFlow Version Check")
    print("=" * 60)
    
    tf_version = tf.__version__
    print(f"TensorFlow version: {tf_version}")
    
    # Check if version is compatible (2.10+)
    major, minor = map(int, tf_version.split('.')[:2])
    if major >= 2 and minor >= 10:
        print("✓ TensorFlow version is compatible with modern distributed training")
    else:
        print("✗ TensorFlow version may have compatibility issues")
        return False
    
    return True

def test_gpu_setup():
    """Test GPU configuration."""
    print("\n" + "=" * 60)
    print("GPU Configuration Test")
    print("=" * 60)
    
    # List GPUs
    gpus = tf.config.list_physical_devices('GPU')
    print(f"Found {len(gpus)} GPU(s)")
    
    if not gpus:
        print("✗ No GPUs found - multi-GPU training not possible")
        return False
    
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
    
    # Test memory growth setting
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ Memory growth enabled successfully")
    except RuntimeError as e:
        print(f"✗ Memory growth setting failed: {e}")
        return False
    
    return len(gpus) > 1

def test_mirrored_strategy():
    """Test MirroredStrategy functionality."""
    print("\n" + "=" * 60)
    print("MirroredStrategy Test")
    print("=" * 60)
    
    try:
        strategy = tf.distribute.MirroredStrategy()
        print(f"✓ MirroredStrategy created with {strategy.num_replicas_in_sync} replicas")
        
        # Test variable creation
        with strategy.scope():
            var = tf.Variable(2.0, name="test_var")
            print("✓ Variable creation in strategy scope successful")
        
        # Test reduction
        @tf.function
        def test_reduction():
            return strategy.reduce(tf.distribute.ReduceOp.SUM, var, axis=None)
        
        result = test_reduction()
        expected = 2.0 * strategy.num_replicas_in_sync
        print(f"✓ Reduction test: {result.numpy()} (expected: {expected})")
        
        if abs(result.numpy() - expected) < 1e-6:
            print("✓ Reduction working correctly")
        else:
            print("✗ Reduction result unexpected")
            return False
            
    except Exception as e:
        print(f"✗ MirroredStrategy test failed: {e}")
        return False
    
    return True

def test_distributed_dataset():
    """Test distributed dataset functionality."""
    print("\n" + "=" * 60)
    print("Distributed Dataset Test")
    print("=" * 60)
    
    try:
        strategy = tf.distribute.MirroredStrategy()
        
        # Create dummy dataset
        def make_dataset():
            x = tf.random.normal((100, 32, 32, 1))
            y = tf.random.uniform((100,), maxval=10, dtype=tf.int32)
            dataset = tf.data.Dataset.from_tensor_slices((x, y))
            dataset = dataset.batch(8, drop_remainder=True)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            return dataset
        
        dataset = make_dataset()
        
        # Test old API (still supported in TF 2.12 but deprecated)
        dist_dataset = strategy.experimental_distribute_dataset(dataset)
        print("✓ experimental_distribute_dataset works")
        
        # Test getting one batch
        for batch in dist_dataset.take(1):
            if hasattr(batch, 'values'):
                batch_x, batch_y = batch
                print(f"✓ Got distributed batch with {len(batch_x.values)} replicas")
                print(f"  Per-replica batch shape: {batch_x.values[0].shape}")
            else:
                print("✓ Got regular batch (single device)")
                break
        
    except Exception as e:
        print(f"✗ Distributed dataset test failed: {e}")
        return False
    
    return True

def test_model_compilation():
    """Test model compilation in distributed scope."""
    print("\n" + "=" * 60)
    print("Model Compilation Test")
    print("=" * 60)
    
    try:
        strategy = tf.distribute.MirroredStrategy()
        
        with strategy.scope():
            # Simple model for testing
            model = keras.Sequential([
                keras.layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 1)),
                keras.layers.GlobalAveragePooling2D(),
                keras.layers.Dense(10, activation='softmax')
            ])
            
            optimizer = keras.optimizers.Adam(learning_rate=0.001)
            model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            print("✓ Model compiled successfully in strategy scope")
            print(f"  Model has {model.count_params()} parameters")
        
    except Exception as e:
        print(f"✗ Model compilation test failed: {e}")
        return False
    
    return True

def test_custom_training_step():
    """Test custom training step with distributed training."""
    print("\n" + "=" * 60)
    print("Custom Training Step Test")
    print("=" * 60)
    
    try:
        strategy = tf.distribute.MirroredStrategy()
        
        class TestModel(keras.Model):
            def __init__(self):
                super().__init__()
                self.conv1 = keras.layers.Conv2D(32, 3, activation='relu')
                self.pool = keras.layers.GlobalAveragePooling2D()
                self.dense = keras.layers.Dense(10)
                self.loss_tracker = keras.metrics.Mean(name='loss')
            
            @property
            def metrics(self):
                return [self.loss_tracker]
            
            def call(self, inputs):
                x = self.conv1(inputs)
                x = self.pool(x)
                return self.dense(x)
            
            @tf.function
            def train_step(self, data):
                x, y = data
                with tf.GradientTape() as tape:
                    predictions = self(x, training=True)
                    loss = self.compiled_loss(y, predictions)
                
                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                
                self.loss_tracker.update_state(loss)
                return {m.name: m.result() for m in self.metrics}
        
        with strategy.scope():
            model = TestModel()
            model.compile(
                optimizer=keras.optimizers.Adam(),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            )
            
            # Test with dummy data
            x = tf.random.normal((16, 32, 32, 1))
            y = tf.random.uniform((16,), maxval=10, dtype=tf.int32)
            
            # Single training step
            result = model.train_step((x, y))
            print(f"✓ Custom training step executed: loss = {result['loss']:.4f}")
        
    except Exception as e:
        print(f"✗ Custom training step test failed: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("TensorFlow 2.12 Multi-GPU Compatibility Test")
    print("=" * 60)
    
    tests = [
        ("TensorFlow Version", test_tf_version),
        ("GPU Setup", test_gpu_setup),
        ("MirroredStrategy", test_mirrored_strategy),
        ("Distributed Dataset", test_distributed_dataset),
        ("Model Compilation", test_model_compilation),
        ("Custom Training Step", test_custom_training_step),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:.<30} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("\n✓ All tests passed! Multi-GPU training should work correctly.")
        print("\nNext steps:")
        print("  1. Run: python run.py --config configs/config_multi_gpu.yaml --training --multi_gpu")
        print("  2. Monitor GPU utilization with: nvidia-smi")
    else:
        print("\n✗ Some tests failed. Check the issues above before using multi-GPU training.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
