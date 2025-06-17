import numpy as np
import tensorflow as tf
from tensorflow import keras

from diffusion_utils import DiffusionUtility
from layers import (
    kernel_init,
    TimeEmbedding,
    TimeMLP,
    SpaceToDepthLayer,
    DepthToSpaceLayer,
    ResidualBlock,
    DownSample,
    UpSample,
)
from unet import build_model
from diffusion_model import DiffusionModel

__all__ = [
    'DiffusionUtility',
    'kernel_init',
    'TimeEmbedding',
    'TimeMLP',
    'SpaceToDepthLayer',
    'DepthToSpaceLayer',
    'ResidualBlock',
    'DownSample',
    'UpSample',
    'build_model',
    'DiffusionModel',
]

if __name__ == "__main__":
    model = build_model()
    model.summary()
    model.save("saved_model_tf" + tf.__version__ + ".h5", include_optimizer=False)
    loaded_model = keras.models.load_model(
        "saved_model_tf" + tf.__version__ + ".h5",
        custom_objects={
            "TimeEmbedding": TimeEmbedding,
            "SpaceToDepthLayer": SpaceToDepthLayer,
            "DepthToSpaceLayer": DepthToSpaceLayer,
        },
    )
    x = np.random.randn(5, 256, 256, 1).astype(np.float32)
    t = np.random.randint(0, 100, 5, dtype=np.int32)
    y_pred = loaded_model.predict([x, t])
    print(y_pred.shape)

