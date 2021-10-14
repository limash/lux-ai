import pickle
import glob
import random
# from pathlib import Path

# import reverb

from lux_ai import dm_reverb_storage, collector, scraper, trainer, tools
from lux_gym.envs.lux.action_vectors import actions_number
from run_configuration import CONF_Scraper, CONF_Single, CONF_ActorCritic

main_config = CONF_ActorCritic
single_config = CONF_Single
scraper_config = CONF_Scraper
# complex_config = CONF_Complex


def imitate(input_data):  # , checkpoint):
    config = {**main_config, **single_config}
    # if checkpoint is not None:
    #     path = str(Path(checkpoint).parent)  # due to https://github.com/deepmind/reverb/issues/12
    #     checkpointer = reverb.checkpointers.DefaultCheckpointer(path=path)
    # else:
    #     checkpointer = None

    # feature_maps_shape = tools.get_feature_maps_shape(config["environment"])
    # buffer = storage.UniformBuffer(feature_maps_shape,
    #                                num_tables=1, min_size=config["batch_size"], max_size=config["buffer_size"],
    #                                n_points=config["n_points"], checkpointer=checkpointer)
    # init collector:
    # collector_agent = collector.Agent(config)
    # collector_agent.collect_and_store(2)
    # init trainer
    trainer_agent = trainer.Agent(config, input_data)
    trainer_agent.imitate()
    # init single_agent, which collects and trains

    # data = {
    #     'weights': weights,
    #     'mask': mask,
    #     'reward': reward
    # }
    # with open('data/data.pickle', 'wb') as f:
    #     pickle.dump(data, f, protocol=4)
    # with open('data/checkpoint', 'w') as text_file:
    #     print(checkpoint, file=text_file)
    print("Done")


def scrape():
    import ray

    config = {**main_config, **scraper_config}

    if config["scrape_type"] == "single":
        scraper_agent = scraper.Agent(config)
        scraper_agent.scrape_all()
    elif config["scrape_type"] == "multi":
        parallel_calls = 8
        env_name = config["environment"]
        feature_maps_shape = tools.get_feature_maps_shape(config["environment"])
        lux_version = config["lux_version"]
        team_name = config["team_name"]
        only_wins = config["only_wins"]
        files_pool = set(glob.glob("./data/jsons/*.json"))
        set_size = int(len(files_pool) / parallel_calls)
        sets = []
        for _ in range(parallel_calls):
            new_set = set(random.sample(files_pool, set_size))
            files_pool -= new_set
            sets.append(new_set)

        already_saved_files = glob.glob("./data/tfrecords/imitator/train/*.tfrec")

        for i, file_names in enumerate(zip(*sets)):
            print(f"Iteration {i} starts")
            ray.init(num_cpus=parallel_calls, num_gpus=1)
            scraper_object = ray.remote(num_gpus=1 / parallel_calls)(scraper.scrape_file)
            futures = [scraper_object.remote(env_name, file_names[j], team_name,
                                             already_saved_files, lux_version, only_wins,
                                             feature_maps_shape, actions_number, i)
                       for j in range(len(file_names))]
            _ = ray.get(futures)
            ray.shutdown()
            # [ray.cancel(object_ref) for object_ref in futures]
    else:
        raise ValueError

    return 0


if __name__ == '__main__':
    try:
        with open('data/data.pickle', 'rb') as file:
            init_data = pickle.load(file)
    except FileNotFoundError:
        init_data = None

    # init_checkpoint = None  # "./data/dmreverb_store/looking_for_halite/checkpoint"

    if main_config["setup"] == "imitate":
        imitate(init_data)  # , init_checkpoint)
    elif main_config["setup"] == "scrape":
        scrape()
    else:
        raise NotImplementedError
