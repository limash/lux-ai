CONF_ActorCritic = {
    "environment": "lux_gym:lux-v0",
    "setup": "single",
    "lambda": 0.8,
    "saved_policies": ["ilia_rb"]
}

CONF_Single = {
    "debug": True,
    "default_lr": 1e-8,
    "n_points": 40,
    "buffer_size": 1000000,
    "batch_size": 40,
    "init_episodes": 25,
    "iterations_number": 1000,
    "save_interval": 100,
    "entropy_c": 2.e-3,
    "entropy_c_decay": 0.3,
}

CONF_Complex = {
    "debug": False,
    "collectors": 1,
    "default_lr": 1e-8,
    "n_points": 40,
    "buffer_size": 3000000,
    "batch_size": 100,
    "iterations_number": 10000,
    "save_interval": 1000,
    "entropy_c": 2.e-3,
    "entropy_c_decay": 0.3,
}
