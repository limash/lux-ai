import pickle
import glob
import random
import pathlib

import ray

from lux_ai import scraper, collector, evaluator, trainer, tools
from lux_gym.envs.lux.action_vectors_new import empty_worker_action_vectors
from run_configuration import CONF_Scrape, CONF_RL, CONF_Main, CONF_Imitate, CONF_Evaluate


def scrape():

    config = {**CONF_Main, **CONF_Scrape}

    if config["scrape_type"] == "single":
        scraper_agent = scraper.Agent(config)
        scraper_agent.scrape_all()
    elif config["scrape_type"] == "multi":
        parallel_calls = config["parallel_calls"]
        actions_shape = [item.shape for item in empty_worker_action_vectors]
        env_name = config["environment"]
        feature_maps_shape = tools.get_feature_maps_shape(config["environment"])
        lux_version = config["lux_version"]
        team_name = config["team_name"]
        only_wins = config["only_wins"]
        is_for_rl = config["is_for_rl"]
        files_pool = set(glob.glob("./data/jsons/*.json"))
        if is_for_rl:
            data_path = "./data/tfrecords/rl/storage/"
        else:
            data_path = "./data/tfrecords/imitator/train/"
        already_saved_files = glob.glob(data_path + "*.tfrec")
        file_names_saved = []
        for i, file_name in enumerate(files_pool):
            raw_name = pathlib.Path(file_name).stem
            if f"{data_path}{raw_name}_{team_name}.tfrec" in already_saved_files:
                print(f"File {file_name} for {team_name}; is already saved.")
                file_names_saved.append(file_name)
        files_pool -= set(file_names_saved)

        set_size = int(len(files_pool) / parallel_calls)
        sets = []
        for _ in range(parallel_calls):
            new_set = set(random.sample(files_pool, set_size))
            files_pool -= new_set
            sets.append(new_set)

        for i, file_names in enumerate(zip(*sets)):
            print(f"Iteration {i} starts")
            ray.init(num_cpus=parallel_calls, include_dashboard=False)
            scraper_object = ray.remote(scraper.scrape_file)
            futures = [scraper_object.remote(env_name, file_names[j], team_name, lux_version, only_wins,
                                             feature_maps_shape, actions_shape, i, is_for_rl)
                       for j in range(len(file_names))]
            _ = ray.get(futures)
            ray.shutdown()
    else:
        raise ValueError

    return 0


def collect(input_data):
    config = {**CONF_Main}
    collect_agent = collector.Agent(config, input_data)
    collect_agent.collect_and_store(1)


def evaluate(input_data):
    config = {**CONF_Main, **CONF_Evaluate}
    eval_agent = evaluator.Agent(config, input_data)
    eval_agent.evaluate()


def imitate(input_data):
    config = {**CONF_Main, **CONF_Imitate, **CONF_Evaluate}

    if config["with_evaluation"]:
        ray.init(num_gpus=1, include_dashboard=False)
        # remote objects creation
        trainer_object = ray.remote(num_gpus=1)(trainer.Agent)
        eval_object = ray.remote(evaluator.Agent)
        # initialization
        workers_info = tools.GlobalVarActor.remote()
        trainer_agent = trainer_object.remote(config, input_data, workers_info)
        eval_agent = eval_object.remote(config, input_data, workers_info)
        # remote call
        trainer_future = trainer_agent.imitate.remote()
        eval_future = eval_agent.evaluate.remote()
        # getting results from remote functions
        _ = ray.get(trainer_future)
        _ = ray.get(eval_future)
        ray.shutdown()

    else:
        trainer_agent = trainer.Agent(config, input_data)
        trainer_agent.imitate()


def rl_train(input_data):  # , checkpoint):
    config = {**CONF_Main, **CONF_RL}
    # if checkpoint is not None:
    #     path = str(Path(checkpoint).parent)  # due to https://github.com/deepmind/reverb/issues/12
    #     checkpointer = reverb.checkpointers.DefaultCheckpointer(path=path)
    # else:
    #     checkpointer = None
    # feature_maps_shape = tools.get_feature_maps_shape(config["environment"])
    # buffer = storage.UniformBuffer(feature_maps_shape,
    #                                num_tables=1, min_size=config["batch_size"], max_size=config["buffer_size"],
    #                                n_points=config["n_points"], checkpointer=checkpointer)
    trainer_agent = trainer.ACAgent(config, input_data)
    trainer_agent.do_train()


if __name__ == '__main__':
    try:
        with open('data/data.pickle', 'rb') as file:
            init_data = pickle.load(file)
    except FileNotFoundError:
        init_data = None

    if CONF_Main["setup"] == "scrape":
        scrape()
    elif CONF_Main["setup"] == "collect":
        collect(init_data)
    elif CONF_Main["setup"] == "evaluate":
        evaluate(init_data)
    elif CONF_Main["setup"] == "imitate":
        imitate(init_data)
    elif CONF_Main["setup"] == "rl":
        rl_train(init_data)
    else:
        raise NotImplementedError
