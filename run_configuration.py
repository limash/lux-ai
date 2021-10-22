CONF_Main = {
    "environment": "lux_gym:lux-v0",
    "setup": "imitate",
    "model_name": "actor_critic_residual",
    "n_points": 40,
}

CONF_Scraper = {
    "lux_version": "3.1.0",
    "scrape_type": "single",
    "is_for_rl": False,
    "team_name": "Toad Brigade",
    "only_wins": False,
}

CONF_Imitate = {
    "batch_size": 300,
}

CONF_RL = {
    "lambda": 0.8,
    "debug": True,
    "default_lr": 1e-6,
    "batch_size": 5,
    "iterations_number": 1000,
    "save_interval": 100,
    "entropy_c": 2.e-3,
    "entropy_c_decay": 0.3,
}
