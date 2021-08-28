import pickle
from pathlib import Path

import reverb

from lux_ai import storage, collector
from run_configuration import *

main_config = CONF_ActorCritic
single_config = CONF_Single
complex_config = CONF_Complex


def one_call(input_data, checkpoint):
    config = {**main_config, **single_config}
    if checkpoint is not None:
        path = str(Path(checkpoint).parent)  # due to https://github.com/deepmind/reverb/issues/12
        checkpointer = reverb.checkpointers.DefaultCheckpointer(path=path)
    else:
        checkpointer = None

    buffer = storage.UniformBuffer(num_tables=1,
                                   min_size=config["batch_size"], max_size=config["buffer_size"],
                                   checkpointer=checkpointer)
    # init collector:
    collector_agent = collector.Agent(config, buffer.table_names, buffer.server_port)
    collector_agent.collect_once()
    # init trainer
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
    # print("Done")


if __name__ == '__main__':
    try:
        with open('data/data.pickle', 'rb') as file:
            init_data = pickle.load(file)
    except FileNotFoundError:
        init_data = None

    try:
        init_checkpoint = open('data/checkpoint', 'r').read()
    except FileNotFoundError:
        init_checkpoint = None

    if main_config["setup"] == "single":
        one_call(init_data, init_checkpoint)
    else:
        raise NotImplementedError
