CONF_ActorCritic = {
    "environment": "lux_gym:lux-v0",
    "setup": "scrape",
    # "lambda": 0.8,
    # "saved_policies": ["ilia_rb"],
    "model_name": "actor_critic_efficient",
}

CONF_Scraper = {
    "lux_version": "3.1.0",
    "scrape_type": "single",
    "is_for_rl": True,
    "team_name": "Toad Brigade",
    "only_wins": False,
}

CONF_Single = {
    # "debug": True,
    # "default_lr": 1e-8,
    # "n_points": 40,
    # "buffer_size": 1000000,
    "batch_size": 110,
    # "init_episodes": 25,
    # "iterations_number": 1000,
    # "save_interval": 100,
    # "entropy_c": 2.e-3,
    # "entropy_c_decay": 0.3,
}
