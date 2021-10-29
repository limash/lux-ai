import abc
import glob
import time
import pickle
import pathlib

import tensorflow as tf
import kaggle_environments as kaggle
import ray

import lux_gym.agents.agents as agents

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Agent(abc.ABC):
    def __init__(self, config, data, global_var_actor=None):
        self._model_name = config["model_name"]
        self._compare_agent = agents.get_agent(config["eval_compare_agent"], is_gym=False)
        if data:
            self._agent = agents.get_agent(self._model_name, data, is_gym=False)
        else:
            self._agent = None

        self._global_var_actor = global_var_actor if global_var_actor else None

    def evaluate(self):
        prev_files_n = 0
        summary = [0, 0]

        while True:
            files = glob.glob("./data/weights/*.pickle")
            files_n = len(files)
            if files_n > prev_files_n:
                with open(files[-1], 'rb') as file:
                    data = pickle.load(file)
                self._agent = agents.get_agent(self._model_name, data, is_gym=False)
                summary = [0, 0]
                raw_name = pathlib.Path(files[-1]).stem
                print(f"Evaluating {raw_name}.pickle weights.")
            if self._agent is not None:
                environment = kaggle.make("lux_ai_2021", configuration={"loglevel": 2}, debug=False)
                steps = environment.run([self._compare_agent, self._agent])
                if steps[-1][0]['reward'] > steps[-1][1]['reward']:
                    summary[0] += 1
                elif steps[-1][0]['reward'] < steps[-1][1]['reward']:
                    summary[1] += 1
                else:
                    summary[0] += 1
                    summary[1] += 1
                print(f"Score: {summary}")
            else:
                print("Waiting for data.")
                time.sleep(60)
            if self._global_var_actor is not None:
                is_done = ray.get(self._global_var_actor.get_done.remote())
                if is_done:
                    break

            prev_files_n = files_n

        print("Evaluation is done.")
        time.sleep(1)
