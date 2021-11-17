CONF_Main = {
    "environment": "lux_gym:lux-v0",
    "setup": "imitate",
    "model_name": "actor_critic_efficient_six_actions",
    "n_points": 40,  # check tfrecords reading transformation merge_rl
}

CONF_Scrape = {
    "lux_version": "3.1.0",
    "scrape_type": "multi",
    "parallel_calls": 8,
    "is_for_rl": False,
    "is_pg_rl": False,
    "team_name": None,  # "Toad Brigade",
    "only_wins": True,
    "only_top_teams": False,
}

CONF_Collect = {
    "is_for_imitator": False,
    "is_for_rl": True,
    "is_pg_rl": True,
    "only_wins": False,
}

CONF_Evaluate = {
    "eval_compare_agent": "compare_agent_eff",
}

CONF_Imitate = {
    "batch_size": 500,
    "self_imitation": False,
    "with_evaluation": False,
}

CONF_RL = {
    "rl_type": "from_scratch_pg",
    # "lambda": 0.8,
    "debug": False,
    "default_lr": 1e-5,
    "batch_size": 500,
    # "iterations_number": 1000,
    # "save_interval": 100,
    "entropy_c": 1.e-4,
    "entropy_c_decay": 0.3,
}
