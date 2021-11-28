CONF_Main = {
    "environment": "lux_gym:lux-v0",
    "setup": "imitate",
    "model_name": "actor_critic_sep_residual_six_actions",
    "n_points": 40,  # check tfrecords reading transformation merge_rl
}

CONF_Scrape = {
    "lux_version": "3.1.0",
    "scrape_type": "multi",
    "parallel_calls": 8,
    "is_for_rl": True,
    "is_pg_rl": True,
    "team_name": None,  # "Toad Brigade",
    "only_wins": False,
    "only_top_teams": False,
}

CONF_Collect = {
    "is_for_imitator": True,
    "is_for_rl": False,
    "is_pg_rl": False,
    "only_wins": False,
}

CONF_Evaluate = {
    "eval_compare_agent": "compare_agent_sub13",
}

CONF_Imitate = {
    "batch_size": 300,
    "self_imitation": False,
    "with_evaluation": True,
}

CONF_RL = {
    "rl_type": "continuous_ac_mc",
    # "lambda": 0.8,
    "debug": False,
    "default_lr": 1e-4,
    "batch_size": 400,
    # "iterations_number": 1000,
    # "save_interval": 100,
    "entropy_c": 5.e-3,
    "entropy_c_decay": 0.3,
}
