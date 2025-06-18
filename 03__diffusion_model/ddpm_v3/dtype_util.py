import tensorflow as tf

def get_compute_dtype():
    """Return current compute dtype from the global mixed precision policy."""
    policy = tf.keras.mixed_precision.global_policy()
    return tf.as_dtype(policy.compute_dtype if policy else tf.float32)
