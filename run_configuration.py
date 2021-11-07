CONF_Main = {
    "environment": "lux_gym:lux-v0",
    "setup": "scrape",
    "model_name": "actor_critic_residual_six_actions",
    "n_points": 40,  # check tfrecords reading transformation merge_rl
}

CONF_Scrape = {
    "lux_version": "3.1.0",
    "scrape_type": "single",
    "parallel_calls": 8,
    "is_for_rl": True,
    "team_name": "Toad Brigade",
    "only_wins": False,
}

CONF_Evaluate = {
    "eval_compare_agent": "compare_agent",
}

CONF_Imitate = {
    "batch_size": 600,
    "with_evaluation": True,
}

CONF_RL = {
    "lambda": 0.8,
    "debug": True,
    "default_lr": 1e-6,
    "batch_size": 5,
    # "iterations_number": 1000,
    # "save_interval": 100,
    "entropy_c": 2.e-3,
    "entropy_c_decay": 0.3,
}
