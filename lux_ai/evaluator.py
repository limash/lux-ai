import abc

import tensorflow as tf
import kaggle_environments as kaggle

import lux_gym.agents.agents as agents

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Agent(abc.ABC):
    def __init__(self, config, data):
        self._compare_agent = agents.get_agent(config["compare_agent"], is_gym=False)
        self._agent = agents.get_agent("half_imitator", data, is_gym=False)

    def evaluate(self):
        summary = [0, 0]

        environment = kaggle.make("lux_ai_2021", configuration={"loglevel": 2}, debug=False)
        steps = environment.run([self._compare_agent, self._agent])
        if steps[-1][0]['reward'] > steps[-1][1]['reward']:
            summary[0] += 1
        elif steps[-1][0]['reward'] < steps[-1][1]['reward']:
            summary[1] += 1
        else:
            summary[0] += 1
            summary[1] += 1
        print(summary)
