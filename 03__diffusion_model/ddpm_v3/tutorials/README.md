# DDPM v3 Project

## Overview

The DDPM v3 project is a state-of-the-art implementation of Denoising Diffusion Probabilistic Models (DDPM) for generative modeling. This project provides a comprehensive, production-ready framework for training and generating high-quality images using diffusion models with advanced features like DDIM sampling, class conditioning, and multiple prediction parameterizations.

## Key Features

- **Advanced Diffusion Models**: Full implementation of DDPM with support for noise, image, and velocity prediction parameterizations
- **DDIM Sampling**: Deterministic sampling with configurable eta parameter for faster inference
- **Class Conditioning**: Support for conditional generation with class labels
- **Multiple Noise Schedules**: Linear, cosine, and cosine^6 noise scheduling options
- **EMA Training**: Exponential Moving Average for stable training and better generation quality
- **Modular Architecture**: Clean separation of concerns with dedicated modules for diffusion utilities, U-Net, data loading, and generation
- **Production Features**: Multi-GPU support, XLA compilation, comprehensive logging, and memory monitoring
- **Flexible Configuration**: YAML-based configuration system for easy experimentation
- **Advanced U-Net**: Configurable U-Net with attention mechanisms, residual blocks, and time embeddings

## Architecture Overview

The codebase is organized into several key components:

- **`diffusion_utils.py`**: Core diffusion mathematics and noise scheduling
- **`diffusion_model.py`**: Main training model with EMA and loss tracking
- **`unet.py`**: Configurable U-Net architecture with attention
- **`layers.py`**: Custom layers including time embeddings and residual blocks
- **`image_generator.py`**: Inference engine for image generation
- **`data_loader.py`**: Efficient data loading and preprocessing
- **`run.py`**: Main training and inference script
- **`callbacks.py`**: Training callbacks for logging and monitoring

## Installation

### Prerequisites
- Python 3.8+
- TensorFlow 2.12+
- NumPy, SciPy, tqdm, PyYAML

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ddpm_v3
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow>=2.12.0 numpy scipy tqdm pyyaml pillow matplotlib
   ```

3. Verify GPU setup (optional but recommended):
   ```bash
   python check_gpu.py
   ```

## Quick Start

### Training a Model

1. **Prepare your dataset**: Organize images in a directory or create `.npz` files with image arrays
2. **Configure training**: Copy and modify `configs/config_simple01.yaml`
3. **Start training**:
   ```bash
   python run.py --config configs/your_config.yaml --training
   ```

### Generating Images

1. **Use a trained model**:
   ```bash
   python run.py --config configs/your_config.yaml --imgen
   ```

2. **Quick generation script**:
   ```python
   from diffusion_model import DiffusionModel
   from diffusion_utils import DiffusionUtility
   from unet import build_model
   
   # Load trained model
   model = keras.models.load_model("path/to/trained_model.h5")
   
   # Generate images
   diffusion = DiffusionUtility(timesteps=1000)
   ddpm = DiffusionModel(model, model, diffusion, num_classes=None)
   images = ddpm.sample_images(num_images=16)
   ```

## Documentation

- **[Getting Started Guide](docs/getting-started.html)**: Detailed setup and first steps
- **[Configuration Reference](docs/configuration.html)**: Complete parameter documentation
- **[Training Guide](docs/training.html)**: Training workflows and best practices
- **[Generation Guide](docs/generation.html)**: Image generation and sampling options
- **[API Reference](docs/api-reference.html)**: Complete code documentation
- **[Examples](examples/)**: Practical usage examples

## Advanced Features

### Prediction Parameterizations
The model supports three prediction types:
- **Noise prediction**: Predicts the noise ε added at each timestep (original DDPM)
- **Image prediction**: Directly predicts the clean image x₀
- **Velocity prediction**: Predicts the velocity v_t = α_t ε - σ_t x₀ (often more stable)

### Sampling Methods
- **DDPM**: Full reverse process with T steps
- **DDIM**: Accelerated deterministic sampling with configurable steps
- **Conditional**: Class-conditional generation with learned embeddings

### Multi-GPU Training
```bash
python run.py --config config_multi_gpu.yaml --training
```

## Project Structure

```
ddpm_v3/
├── run.py                 # Main entry point
├── diffusion_model.py     # Core training model
├── diffusion_utils.py     # Diffusion mathematics
├── unet.py               # U-Net architecture
├── layers.py             # Custom layers
├── image_generator.py    # Generation engine
├── data_loader.py        # Data handling
├── callbacks.py          # Training callbacks
├── configs/              # Configuration files
├── tutorials/            # Documentation
└── training_outputs/     # Training results
```

## Getting Started

For a detailed guide on setting up the project, refer to the [Getting Started](docs/getting-started.html) section of the documentation.

## Configuration

The project offers various configuration options. You can find detailed explanations of each parameter in the [Configuration](docs/configuration.html) section.

## Training

To learn how to train the model, including data preparation and hyperparameter settings, visit the [Training](docs/training.html) section.

## Image Generation

Once your model is trained, you can generate images. Instructions on how to do this can be found in the [Generation](docs/generation.html) section.

## API Reference

For developers looking to dive deeper, the [API Reference](docs/api-reference.html) provides detailed information on the available functions and classes.

## Contributing

Contributions are welcome! If you would like to contribute to the DDPM v3 project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch and create a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact

For questions or feedback, please reach out to the project maintainers at [your-email@example.com].