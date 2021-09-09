import abc
import random

# import numpy as np
import tensorflow as tf
import gym
# import reverb
# import ray

from lux_ai import tools, tfrecords_storage
import lux_gym.agents.agents as agents
from lux_gym.envs.lux.action_vectors import action_vector

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


# def get_policy(model):
#     def policy():
#         actions = []
#         return actions
#
#     return policy


class Agent(abc.ABC):

    def __init__(self, config):
        # buffer_table_names, buffer_server_port,
        # data=None,
        # ray_queue=None, collector_id=None, workers_info=None, num_collectors=None
        # ):
        """
        Args:
            config: A configuration dictionary
            # buffer_table_names: dm reverb server table names
            # buffer_server_port: a port where a dm reverb server was initialized
            # data: a neural net weights
            # ray_queue: a ray interprocess queue to store neural net weights
            # collector_id: to identify a current collector if there are several ones
            # workers_info: a ray interprocess (remote) object to store shared information
            # num_collectors: a total amount of collectors
        """
        self._n_players = 2
        self._env_name = config["environment"]

        self._feature_maps_shape = tools.get_feature_maps_shape(config["environment"])
        self._actions_number = len(action_vector)

        self._n_points = config["n_points"]

        # self._table_names = buffer_table_names
        # self._client = reverb.Client(f'localhost:{buffer_server_port}')

        # self._ray_queue = ray_queue
        # self._collector_id = collector_id
        # self._workers_info = workers_info
        # self._num_collectors = num_collectors

        self._policies_pool = [agents.get_processing_agent(name) for name in config["saved_policies"]]

        # if not config["debug"]:
        #     self._predict = tf.function(self._predict)

        # if data:
        #     self._model = models.get_actor_critic2(model_type='exp')
        #     dummy_input = tf.ones(self._feature_maps_shape, dtype=tf.float16)
        #     dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
        #     self._predict(dummy_input)
        #     self._model.set_weights(data['weights'])
        #     self._policy = get_policy(self._predict)

    def _collect(self, agent):
        """
        Collects trajectories from an episode. Episodes consists of n_points.

        One n_point contains (action, action_probs, action_mask, observation,
                              total reward, temporal_mask, progress);
        action is a response for the current observation,
        reward, done are for the current observation.
        """

        player1_data = {}
        player2_data = {}

        environment = gym.make(self._env_name)
        observations = environment.reset()
        configuration = environment.configuration
        game_states = environment.game_states
        actions_1, actions_1_dict, actions_1_probs, proc_obs1, reward1 = agent(observations[0],
                                                                               configuration, game_states[0])
        actions_2, actions_2_dict, actions_2_probs, proc_obs2, reward2 = agent(observations[1],
                                                                               configuration, game_states[1])

        step = 0
        player1_data = tools.add_point(player1_data, actions_1_dict, actions_1_probs, proc_obs1, step)
        player2_data = tools.add_point(player2_data, actions_2_dict, actions_2_probs, proc_obs2, step)

        for step in range(1, configuration.episodeSteps):
            dones, observations = environment.step((actions_1, actions_2))
            game_states = environment.game_states
            actions_1, actions_1_dict, actions_1_probs, proc_obs1, reward1 = agent(observations[0],
                                                                                   configuration, game_states[0])
            actions_2, actions_2_dict, actions_2_probs, proc_obs2, reward2 = agent(observations[1],
                                                                                   configuration, game_states[1])

            player1_data = tools.add_point(player1_data, actions_1_dict, actions_1_probs, proc_obs1, step)
            player2_data = tools.add_point(player2_data, actions_2_dict, actions_2_probs, proc_obs2, step)
            if any(dones):
                break

        progress = tf.linspace(0., 1., step + 2)[:-1]
        progress = tf.cast(progress, dtype=tf.float16)

        if reward1 > reward2:
            final_reward_1 = tf.constant(1, dtype=tf.float16)
            final_reward_2 = tf.constant(-1, dtype=tf.float16)
        elif reward1 < reward2:
            final_reward_2 = tf.constant(1, dtype=tf.float16)
            final_reward_1 = tf.constant(-1, dtype=tf.float16)
        else:
            final_reward_1 = final_reward_2 = tf.constant(0, dtype=tf.float16)

        return (player1_data, player2_data), (final_reward_1, final_reward_2), progress

    def collect_once(self):
        policy = random.sample(self._policies_pool, 1)[0]
        return self._collect(policy)

    def collect_and_store(self, number_of_collects):
        for i in range(number_of_collects):
            (player1_data, player2_data), (final_reward_1, final_reward_2), progress = self.collect_once()
            tfrecords_storage.record_for_imitator(player1_data, player2_data, final_reward_1, final_reward_2,
                                                  self._feature_maps_shape, self._actions_number, i)

# def update_model(self, data):
    #     self._model.set_weights(data['weights'])

    # def add_model_to_pool(self, data):
    #     model = models.get_actor_critic(self._feature_maps_shape, self._actions_number)
    #     dummy_input = tf.ones(self._feature_maps_shape, dtype=tf.float16)
    #     dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
    #     model = tf.function(model)
    #     model(dummy_input)
    #     model.set_weights(data['weights'])
    #     policy = get_policy(model)
    #     self._policies_pool.append(policy)

    # def _predict(self, observation):
    #     return self._model(observation)

    # def _policy(self, current_game_state, observation):
    #     """
    #     Policy method defines response to observation for all units of a player.
    #     :param current_game_state:
    #     :param observation:
    #     :return:
    #     """
    #     actions = []
    #     logits = []
    #     observation = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), observation)
    #     policy_logits, _ = self._predict(observation)
    #     action = tf.random.categorical(policy_logits, num_samples=1, dtype=tf.int32)
    #     actions.append(action.numpy()[0][0])
    #     logits.append(policy_logits.numpy()[0])
    #     # probabilities = tf.nn.softmax(policy_logits)
    #     # return np.argmax(probabilities[0])

    #     return actions  # , actions_dict, actions_probs_dict, processed_observations

    # def do_collect(self):
    #     num_collects = 0
    #     num_updates = 0

    #     while True:
    #         # trainer will switch to done on the last iteration
    #         is_done = ray.get(self._workers_info.get_done.remote())
    #         if is_done:
    #             # print("Collecting is done.")
    #             return num_collects, num_updates
    #         # get the current turn, so collectors (workers) update weights one by one
    #         curr_worker = ray.get(self._workers_info.get_global_v.remote())
    #         # check the current turn
    #         if curr_worker == self._collector_id:
    #             if not self._ray_queue.empty():  # see below
    #                 try:
    #                     # block = False will cause an exception if there is no data in the queue,
    #                     # which is not handled by a ray queue (incompatibility with python 3.8 ?)
    #                     weights = self._ray_queue.get(block=False)
    #                     if curr_worker == self._num_collectors:
    #                         # print(f"Worker {curr_worker} updates weights")
    #                         ray.get(self._workers_info.set_global_v.remote(1))
    #                         num_updates += 1
    #                     elif curr_worker < self._num_collectors:
    #                         ray.get(self._workers_info.set_global_v.remote(curr_worker + 1))
    #                         # print(f"Worker {curr_worker} update weights")
    #                         num_updates += 1
    #                     else:
    #                         print("Wrong worker")
    #                         raise NotImplementedError
    #                 except Empty:
    #                     weights = None
    #             else:
    #                 weights = None
    #         else:
    #             weights = None

    #         if weights is not None:
    #             self._model.set_weights(weights)
    #             # print("Weights are updated")

    #         epsilon = None
    #         # t1 = time.time()
    #         if self._data is not None:
    #             if num_collects % 25 == 0:
    #                 self._collect(epsilon, is_random=True)
    #                 # print("Episode with a random trajectory was collected; "
    #                 #       f"Num of collects: {num_collects}")
    #             else:
    #                 self._collect(epsilon)
    #                 # print(f"Num of collects: {num_collects}")
    #         else:
    #             if num_collects < 10000 or num_collects % 25 == 0:
    #                 if num_collects == 9999:
    #                     print("Collector: The last initial random collect.")
    #                 self._collect(epsilon, is_random=True)
    #             else:
    #                 self._collect(epsilon)
    #         num_collects += 1
    #         # print(f"Num of collects: {num_collects}")
    #         # t2 = time.time()
    #         # print(f"Collecting. Time: {t2 - t1}")
