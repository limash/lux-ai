import abc
import random

import numpy as np
import tensorflow as tf
import gym
import reverb
import ray

from lux_ai import models, tools
import lux_gym.agents.agents as agents


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def get_policy(model):
    def policy():
        actions = []
        return actions

    return policy


class Agent(abc.ABC):

    def __init__(self, config,
                 buffer_table_names, buffer_server_port,
                 # data=None,
                 ray_queue=None, collector_id=None, workers_info=None, num_collectors=None
                 ):
        """
        Args:
            config: A configuration dictionary
            buffer_table_names: dm reverb server table names
            buffer_server_port: a port where a dm reverb server was initialized
            # data: a neural net weights
            ray_queue: a ray interprocess queue to store neural net weights
            collector_id: to identify a current collector if there are several ones
            workers_info: a ray interprocess (remote) object to store shared information
            num_collectors: a total amount of collectors
        """
        self._n_players = 2
        self._environment = gym.make(config["environment"])

        self._feature_maps_shape = tools.get_feature_maps_shape(config["environment"])

        self._n_points = config["n_points"]

        self._table_names = buffer_table_names
        self._client = reverb.Client(f'localhost:{buffer_server_port}')

        self._ray_queue = ray_queue
        self._collector_id = collector_id
        self._workers_info = workers_info
        self._num_collectors = num_collectors

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

    # def update_model(self, data):
    #     self._model.set_weights(data['weights'])

    def add_model_to_pool(self, data):
        model = models.get_actor_critic()
        dummy_input = tf.ones(self._feature_maps_shape, dtype=tf.float16)
        dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
        model = tf.function(model)
        model(dummy_input)
        model.set_weights(data['weights'])
        policy = get_policy(model)
        self._policies_pool.append(policy)

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

    def _collect(self, agent):
        """
        Collects trajectories from an episode to the buffer.

        A buffer contains items, each item consists of several n_points;
        One n_point contains (action, action_probs, action_mask, observation,
                              total reward, temporal_mask, progress);
        action is a response for the current observation,
        reward, done are for the current observation.
        """

        worker_action_mask = np.zeros(39, dtype=np.half)
        worker_action_mask[3:22] = 1
        cart_action_mask = np.zeros(39, dtype=np.half)
        cart_action_mask[22:] = 1
        citytile_action_mask = np.zeros(39, dtype=np.half)
        citytile_action_mask[:3] = 1
        actions_masks = (worker_action_mask, cart_action_mask, citytile_action_mask)

        player1_data = {}
        player2_data = {}

        def add_point(player_data, actions_dict, actions_probs, proc_obs, current_step):
            for i, (acts, acts_prob, obs) in enumerate(zip(actions_dict.values(),
                                                           actions_probs.values(),
                                                           proc_obs.values())):
                for (k, action), (_, action_probs), (_, observation) in zip(acts.items(),
                                                                            acts_prob.items(),
                                                                            obs.items()):
                    value = [action, action_probs, actions_masks[i], observation]
                    if k in player_data.keys():
                        player_data[k].append(value, current_step)
                    else:
                        player_data[k] = tools.DataValue()
                        player_data[k].append(value, current_step)
            return player_data

        observations = self._environment.reset()
        configuration = self._environment.configuration
        game_states = self._environment.game_states
        actions_1, actions_1_dict, actions_1_probs, proc_obs1, reward1 = agent(observations[0],
                                                                               configuration, game_states[0])
        actions_2, actions_2_dict, actions_2_probs, proc_obs2, reward2 = agent(observations[1],
                                                                               configuration, game_states[1])

        step = 0
        player1_data = add_point(player1_data, actions_1_dict, actions_1_probs, proc_obs1, step)
        player2_data = add_point(player2_data, actions_2_dict, actions_2_probs, proc_obs2, step)

        for step in range(1, configuration.episodeSteps):
            dones, observations = self._environment.step((actions_1, actions_2))
            game_states = self._environment.game_states
            actions_1, actions_1_dict, actions_1_probs, proc_obs1, reward1 = agent(observations[0],
                                                                                   configuration, game_states[0])
            actions_2, actions_2_dict, actions_2_probs, proc_obs2, reward2 = agent(observations[1],
                                                                                   configuration, game_states[1])

            player1_data = add_point(player1_data, actions_1_dict, actions_1_probs, proc_obs1, step)
            player2_data = add_point(player2_data, actions_2_dict, actions_2_probs, proc_obs2, step)
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

        obs_zeros = tf.zeros(self._feature_maps_shape, dtype=tf.float16)
        act_zeros = tf.zeros(39, dtype=tf.float16)
        act_ones = tf.ones(39, dtype=tf.float16)
        act_probs_uni = tf.ones(39, dtype=tf.float16) * 1/39

        def send_data(player_data, total_reward):
            for data_object in player_data.values():
                entity_temporal_data_list, current_step = data_object.data, data_object.step
                entity_temporal_data_list = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float16),
                                                                  entity_temporal_data_list)
                with self._client.trajectory_writer(num_keep_alive_refs=self._n_points) as writer:
                    for i, data_entry in enumerate(entity_temporal_data_list):
                        act, act_probs, act_mask, obs = data_entry
                        writer.append({'action': act,
                                       'action_probs': act_probs,
                                       'action_mask': act_mask,
                                       'observation': obs,
                                       'total_reward': total_reward,
                                       'temporal_mask': tf.constant(1, dtype=tf.float16),
                                       'progress': progress[current_step[i]]
                                       })
                        if i >= self._n_points - 1:
                            writer.create_item(
                                table=self._table_names[0],
                                priority=1.5,
                                trajectory={
                                    'actions': writer.history['action'][-self._n_points:],
                                    'actions_probs': writer.history['action_probs'][-self._n_points:],
                                    'actions_masks': writer.history['action_mask'][-self._n_points:],
                                    'observations': writer.history['observation'][-self._n_points:],
                                    'total_rewards': writer.history['total_reward'][-self._n_points:],
                                    'temporal_masks': writer.history['temporal_mask'][-self._n_points:],
                                    'progresses': writer.history['progress'][-self._n_points:],
                                }
                            )
                    i += 1
                    for j in range(i, i + self._n_points - 1):
                        writer.append({'action': act_zeros,
                                       'action_probs': act_probs_uni,
                                       'action_mask': act_ones,
                                       'observation': obs_zeros,
                                       'total_reward': tf.constant(0, dtype=tf.float16),
                                       'temporal_mask': tf.constant(0, dtype=tf.float16),
                                       'progress': tf.constant(1, dtype=tf.float16)
                                       })
                        if j >= self._n_points - 1:
                            writer.create_item(
                                table=self._table_names[0],
                                priority=1.5,
                                trajectory={
                                    'actions': writer.history['action'][-self._n_points:],
                                    'actions_probs': writer.history['action_probs'][-self._n_points:],
                                    'actions_masks': writer.history['action_mask'][-self._n_points:],
                                    'observations': writer.history['observation'][-self._n_points:],
                                    'total_rewards': writer.history['total_reward'][-self._n_points:],
                                    'temporal_masks': writer.history['temporal_mask'][-self._n_points:],
                                    'progresses': writer.history['progress'][-self._n_points:],
                                }
                            )
                    writer.end_episode()

        send_data(player1_data, final_reward_1)
        send_data(player2_data, final_reward_2)

    def collect_once(self):
        policy = random.sample(self._policies_pool, 1)[0]
        self._collect(policy)

    def do_collect(self):
        num_collects = 0
        num_updates = 0

        while True:
            # trainer will switch to done on the last iteration
            is_done = ray.get(self._workers_info.get_done.remote())
            if is_done:
                # print("Collecting is done.")
                return num_collects, num_updates
            # get the current turn, so collectors (workers) update weights one by one
            curr_worker = ray.get(self._workers_info.get_global_v.remote())
            # check the current turn
            if curr_worker == self._collector_id:
                if not self._ray_queue.empty():  # see below
                    try:
                        # block = False will cause an exception if there is no data in the queue,
                        # which is not handled by a ray queue (incompatibility with python 3.8 ?)
                        weights = self._ray_queue.get(block=False)
                        if curr_worker == self._num_collectors:
                            # print(f"Worker {curr_worker} updates weights")
                            ray.get(self._workers_info.set_global_v.remote(1))
                            num_updates += 1
                        elif curr_worker < self._num_collectors:
                            ray.get(self._workers_info.set_global_v.remote(curr_worker + 1))
                            # print(f"Worker {curr_worker} update weights")
                            num_updates += 1
                        else:
                            print("Wrong worker")
                            raise NotImplementedError
                    except Empty:
                        weights = None
                else:
                    weights = None
            else:
                weights = None

            if weights is not None:
                self._model.set_weights(weights)
                # print("Weights are updated")

            epsilon = None
            # t1 = time.time()
            if self._data is not None:
                if num_collects % 25 == 0:
                    self._collect(epsilon, is_random=True)
                    # print("Episode with a random trajectory was collected; "
                    #       f"Num of collects: {num_collects}")
                else:
                    self._collect(epsilon)
                    # print(f"Num of collects: {num_collects}")
            else:
                if num_collects < 10000 or num_collects % 25 == 0:
                    if num_collects == 9999:
                        print("Collector: The last initial random collect.")
                    self._collect(epsilon, is_random=True)
                else:
                    self._collect(epsilon)
            num_collects += 1
            # print(f"Num of collects: {num_collects}")
            # t2 = time.time()
            # print(f"Collecting. Time: {t2 - t1}")
