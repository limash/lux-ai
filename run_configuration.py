import tensorflow as tf
import tensorflow_addons as tfa


CONF_ActorCritic = {
    "environment": "lux_gym:lux-v0",
    "setup": "single",
    "optimizer": tfa.optimizers.AdamW(weight_decay=1.e-5, learning_rate=1.e-6),
    "lambda": tf.constant(.8, dtype=tf.float32),
    "saved_policies": ["ilia_rb"]
}

CONF_Single = {
    "debug": True,
    "default_lr": 1e-8,
    "n_points": 2,
    "buffer_size": 1000000,
    "batch_size": 10,
    "init_episodes": 25,
    "iterations_number": 1000,
    "save_interval": 100,
    "entropy_c": tf.constant(2.e-3),
    "entropy_c_decay": tf.constant(0.3),
}

CONF_Complex = {
    "debug": False,
    "collectors": 1,
    "default_lr": 1e-8,
    "n_points": 33,
    "buffer_size": 3000000,
    "batch_size": 100,
    "iterations_number": 10000,
    "save_interval": 1000,
    "entropy_c": tf.constant(2.e-3),
    "entropy_c_decay": tf.constant(0.3),
}
