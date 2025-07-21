# DDPM v3 Code Analysis and Architecture

## Overview

DDPM v3 is a comprehensive implementation of Denoising Diffusion Probabilistic Models with advanced features including multiple prediction parameterizations, DDIM sampling, class conditioning, and production-ready training infrastructure.

## Architecture Analysis

### 1. Modular Design

The codebase follows a clean separation of concerns:

```
ddpm_v3/
├── diffusion_utils.py      # Core diffusion mathematics
├── diffusion_model.py      # Training wrapper
├── unet.py                 # Neural network architecture  
├── layers.py               # Custom building blocks
├── image_generator.py      # Inference engine
├── data_loader.py          # Data handling
├── callbacks.py            # Training utilities
└── run.py                  # Main interface
```

### 2. Core Components Deep Dive

#### DiffusionUtility (diffusion_utils.py)
**Purpose**: Implements the mathematical foundation of diffusion models.

**Key Features**:
- Multiple noise schedules (linear, cosine, cos6)
- Three prediction parameterizations (noise, image, velocity)
- DDPM and DDIM sampling algorithms
- Numerical stability optimizations

**Mathematical Implementation**:
```python
# Forward process: q(x_t | x_0)
def q_sample(self, x0, t, noise=None):
    # x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε
    
# Reverse process: p(x_{t-1} | x_t)
def p_sample(self, x_t, t, pred, clip_denoise=True):
    # Uses learned prediction to estimate x_{t-1}
```

**Prediction Types**:
1. **Noise**: ε_θ(x_t, t) = ε (original DDPM)
2. **Image**: x_θ(x_t, t) = x_0 (direct image prediction)  
3. **Velocity**: v_θ(x_t, t) = α_t ε - σ_t x_0 (stable parameterization)

#### DiffusionModel (diffusion_model.py)
**Purpose**: Keras training wrapper with EMA and comprehensive loss tracking.

**Key Features**:
- Exponential Moving Average (EMA) for stable inference
- Multi-component loss tracking (noise, image, velocity)
- Automatic gradient clipping
- Modular image generation delegation

**Training Loop**:
```python
@tf.function
def train_step(self, data):
    # 1. Sample random timesteps
    # 2. Add noise to images
    # 3. Predict with U-Net
    # 4. Compute losses for all parameterizations
    # 5. Update weights and EMA
```

#### U-Net Architecture (unet.py)
**Purpose**: Configurable U-Net with modern architectural improvements.

**Design Highlights**:
- Skip connections for multi-scale features
- Sinusoidal time embeddings
- Multi-head self-attention at multiple resolutions
- Group normalization for training stability
- Class conditioning support

**Architecture Flow**:
```
Input Image → Encoder Path → Bottleneck → Decoder Path → Output
    ↓              ↓            ↓            ↑         ↑
Time Emb ─────── Injection ────┴──── Skip Connections ─┘
Class Emb ───────┘
```

#### Custom Layers (layers.py)
**Purpose**: Specialized building blocks optimized for diffusion models.

**Components**:
- **TimeEmbedding**: Sinusoidal positional encoding for timesteps
- **ResidualBlock**: Time-conditioned residual blocks with GroupNorm
- **AttentionBlock**: Multi-head self-attention for spatial features
- **DownSample/UpSample**: Learnable resolution changes

#### ImageGenerator (image_generator.py)
**Purpose**: Separate inference engine for clean separation of training and generation.

**Generation Modes**:
- Unconditional sampling
- Class-conditional generation
- Channel inpainting
- DDPM vs DDIM sampling

#### DataLoader (data_loader.py)
**Purpose**: Efficient data pipeline with TensorFlow optimizations.

**Features**:
- NPZ and directory-based loading
- Automatic train/validation splits
- TensorFlow data pipeline optimizations
- Memory-efficient preprocessing

### 3. Advanced Features

#### Multiple Prediction Parameterizations
The implementation supports three prediction types, allowing for optimal training stability:

```python
# Noise prediction (ε-parameterization)
if pred_type == 'noise':
    loss = MSE(noise, predicted_noise)

# Image prediction (x₀-parameterization) 
elif pred_type == 'image':
    loss = MSE(clean_image, predicted_image)

# Velocity prediction (v-parameterization)
elif pred_type == 'velocity':
    v_target = alpha_t * noise - sigma_t * clean_image
    loss = MSE(v_target, predicted_velocity)
```

#### DDIM Sampling Implementation
Deterministic sampling with configurable step size:

```python
def ddim_step(self, x_t, t, pred, ddim_eta=1.0):
    # Deterministic component
    pred_x0 = self.predict_x0_from_eps(x_t, t, pred)
    
    # Stochastic component (controlled by eta)
    if ddim_eta > 0:
        noise = tf.random.normal(tf.shape(x_t))
        sigma = ddim_eta * sqrt((1-alpha_prev)/(1-alpha_t)) * sqrt(1-alpha_t/alpha_prev)
    
    return pred_x0 * sqrt(alpha_prev) + direction + sigma * noise
```

#### Class Conditioning
Seamless support for conditional generation:

```python
# Embedding layer for class labels
class_emb = Embedding(num_classes, class_emb_dim)(class_input)

# Combine with time embedding
combined_emb = time_emb + class_emb

# Inject into U-Net layers
conditioned_features = ResidualBlock()([features, combined_emb])
```

### 4. Production Features

#### Exponential Moving Average (EMA)
Maintains a slowly-updating copy of model weights for better inference:

```python
@tf.function
def _update_ema_weights(self):
    for ema_weight, weight in zip(self.ema_network.trainable_weights, 
                                  self.network.trainable_weights):
        ema_weight.assign(ema_weight * self.ema + (1 - self.ema) * weight)
```

#### Comprehensive Logging
Detailed metrics tracking and structured output directories:

```
training_outputs/
├── {dataset}_{resolution}/
│   └── {architecture_tag}/
│       └── {scheduler}_{timesteps}_{pred_type}_{timestamp}/
│           ├── train.log
│           ├── log.csv  
│           ├── training_config.yaml
│           ├── model_checkpoints.h5
│           └── inline_gen/
```

#### Memory Management
Built-in memory monitoring and optimization:

```python
class MemoryLogger:
    def log_memory_usage(self, stage):
        # TensorFlow memory stats
        # System memory usage
        # GPU memory utilization
```

### 5. Configuration System

YAML-based configuration with comprehensive validation:

```yaml
DATASET:
  NAME: "my_dataset"
  PATH: "./data/"
  LABEL_KEY: "labels"

TRAINING:
  INPUT_IMAGE_SIZE: 256
  PRED_TYPE: 'velocity'
  
  NETWORK:
    SCHEDULER: 'cosine'
    TIMESTEPS: 1000
    FIRST_CHANNEL: 128
    CHANNEL_MULTIPLIER: [1, 2, 4, 8]
    HAS_ATTENTION: [False, False, True, True]

IMAGE_GENERATION:
  GEN_TASK: "class_cond"
  REVERSE_STRIDE: 50  # DDIM sampling
  DDIM_ETA: 0.0       # Deterministic
```

### 6. Performance Optimizations

#### XLA Compilation Support
```python
# Enable XLA for faster execution
if FLAGS.enable_xla:
    tf.config.optimizer.set_jit(True)
```

#### Mixed Precision Ready
```python
# Supports TensorFlow mixed precision
# Automatic loss scaling for numerical stability
```

#### Multi-GPU Training
```python
# Distributed training support
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model(...)
```

### 7. Code Quality Features

#### Type Hints and Documentation
All functions include comprehensive docstrings and type annotations.

#### Error Handling
Robust parameter validation and informative error messages.

#### Testing Infrastructure
Built-in GPU compatibility checking and model validation.

## Key Innovations

1. **Unified Parameterizations**: Support for all three prediction types in a single codebase
2. **Production Architecture**: Clean separation between training and inference
3. **Advanced Sampling**: Both DDPM and DDIM with configurable determinism
4. **Comprehensive Conditioning**: Support for class labels and future extensibility
5. **Memory Efficiency**: Built-in memory monitoring and optimization strategies
6. **Flexible Configuration**: YAML-based system with extensive validation

## Usage Patterns

### Research Workflow
```python
# Quick experimentation
ddpm = DiffusionModel(network, ema_network, diff_util)
ddpm.fit(dataset, epochs=100)
images = ddpm.sample_images(num_images=64)
```

### Production Workflow  
```python
# Configuration-driven training
python run.py --config production_config.yaml --training
python run.py --config production_config.yaml --imgen
```

### Custom Extensions
```python
# Easy to extend with custom components
class CustomDiffusionModel(DiffusionModel):
    def custom_loss(self, y_true, y_pred):
        # Custom loss implementation
        pass
```

This architecture provides a robust foundation for both research and production use cases while maintaining clarity and extensibility.
