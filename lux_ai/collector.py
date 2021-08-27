import abc
import random
import pickle
import itertools as it
import time

import numpy as np
import tensorflow as tf
import gym
import reverb
import ray

from lux_ai import models
import lux_gym.agents.agents as agents


def get_policy(model):
    def policy():
        actions = []
        return actions

    return policy


class Agent(abc.ABC):

    def __init__(self, config,
                 buffer_table_names, buffer_server_port,
                 data=None,
                 ray_queue=None, collector_id=None, workers_info=None, num_collectors=None
                 ):
        """
        Args:
            config: A configuration dictionary
            buffer_table_names: dm reverb server table names
            buffer_server_port: a port where a dm reverb server was initialized
            data: a neural net weights
            ray_queue: a ray interprocess queue to store neural net weights
            collector_id: to identify a current collector if there are several ones
            workers_info: a ray interprocess (remote) object to store shared information
            num_collectors: a total amount of collectors
        """
        self._n_players = 2
        self._environment = gym.make(config["environment"])

        _, observations = self._environment.reset_process()
        self._feature_maps_shape = observations[0]["workers"]["u_1"].shape

        self._n_points = config["n_points"]

        self._table_names = buffer_table_names
        self._replay_memory_client = reverb.Client(f'localhost:{buffer_server_port}')

        self._ray_queue = ray_queue
        self._worker_id = collector_id
        self._workers_info = workers_info
        self._num_collectors = num_collectors

        self._policies_pool = [agents.get_processing_agent(name) for name in config["saved_policies"]]

        if not config["debug"]:
            self._predict = tf.function(self._predict)

        self._model = models.get_actor_critic2(model_type='exp')
        dummy_input = tf.ones(self._feature_maps_shape, dtype=tf.float16)
        dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
        self._predict(dummy_input)

    def add_current_model(self, data):
        self._model.set_weights(data['weights'])

    def add_policy_to_pool(self, data):
        model = models.get_actor_critic2(model_type='exp')
        dummy_input = tf.ones(self._feature_maps_shape, dtype=tf.float16)
        dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
        model(dummy_input)
        model.set_weights(data['weights'])
        policy = get_policy(model)
        self._policies_pool.append(policy)

    def _predict(self, observation):
        return self._model(observation)

    def _policy(self, current_game_state, observation):
        """
        Policy method defines response to observation for all units of a player.
        :param current_game_state:
        :param observation:
        :return:
        """
        actions = []
        logits = []
        observation = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), observation)
        policy_logits, _ = self._predict(observation)
        action = tf.random.categorical(policy_logits, num_samples=1, dtype=tf.int32)
        actions.append(action.numpy()[0][0])
        logits.append(policy_logits.numpy()[0])
        # probabilities = tf.nn.softmax(policy_logits)
        # return np.argmax(probabilities[0])

        return actions  # , actions_dict, actions_probs_dict, processed_observations

    def _collect(self, policy):
        """
        Collects an episode trajectory (1 item) to a buffer.

        A buffer contains items, each item consists of several n_points;
        for a regular TD update an item should have 2 n_points (a minimum number to form one time step).
        One n_point contains (action, obs, reward, done);
        action, reward, done are for the current observation (or obs);
        e.g. action led to the obs, reward prior to the obs, if is it done at the current obs.

        this implementation creates writers for each player (goose) and stores
        n_points trajectories for all of them

        if epsilon is None assume an off policy gradient method where policy_logits required
        """

        # initialize writers for all players
        # writers = [self._replay_memory_client.writer(max_sequence_length=self._n_points)
        #            for _ in range(self._n_players)]
        writers = [[] for _ in range(self._n_players)]
        ray_writers = [self._replay_memory_client.writer(max_sequence_length=self._n_points)
                       for _ in range(self._n_players)]

        dones = [False for _ in range(self._n_players)]  # for a first check
        ready = [False for _ in range(self._n_players)]
        ready_counter = [0 for _ in range(self._n_players)]

        obs_records = []
        info = None

        # some constants we are going to use repeatedly
        action_negative, reward_zero = tf.constant(-1), tf.constant(0.)
        done_true, done_false = tf.constant(1.), tf.constant(0.)
        policy_logits_zeros = tf.constant([0., 0., 0., 0.])
        rewards_saver = [None, None, None, None]
        obs_zeros = (tf.zeros(self._feature_maps_shape, dtype=tf.uint8),
                     tf.zeros(self._scalar_features_shape, dtype=tf.uint8))

        obsns = self._train_env.reset()
        obsns = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.uint8), obsns)
        for i, writer in enumerate(writers):
            obs = obsns[i][0], obsns[i][1]
            obs_records.append(obs)
            if epsilon is None:
                writer.append((action_negative, policy_logits_zeros, obs, reward_zero, done_false))
            else:
                writer.append((action_negative, obs, reward_zero, done_false))
        step_counter = 1  # start with 1, since we have at least initialization
        # steps_per_worker_counter = [1 for _ in range(self._n_players)]

        while not all(ready):
            if not all(dones):
                if epsilon is None:
                    actions, policy_logits = self._policy(obs_records, is_random)
                    policy_logits = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32),
                                                          policy_logits)
                else:
                    actions = self._policy(obs_records, epsilon, info)

                step_counter += 1
                obsns, rewards, dones, info = self._train_env.step(actions)
                # environment step receives actions and outputs observations for the dead players also
                # but it takes no effect
                actions = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.int32), actions)
                rewards = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), rewards)
                dones = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float32), dones)
                obsns = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.uint8), obsns)

            obs_records = []
            for i, writer in enumerate(writers):
                if ready_counter[i] == self._n_points - 1:
                    ready[i] = True
                    obs_records.append(obs_zeros)
                    continue
                action, reward, done = actions[i], rewards[i], dones[i]
                if done:
                    ready_counter[i] += 1
                    # the first 'done' encounter, save a final reward
                    if rewards_saver[i] is None:
                        rewards_saver[i] = reward
                        # steps_per_worker_counter[i] += 1
                    # consequent 'done' encounters, put zero actions and logits
                    else:
                        action = action_negative
                        if epsilon is None:
                            policy_logits[i] = policy_logits_zeros
                    obs = obs_zeros
                    # if 'done', store final rewards
                    reward = rewards_saver[i]
                else:
                    obs = obsns[i][0], obsns[i][1]
                    # steps_per_worker_counter[i] += 1
                obs_records.append(obs)
                if epsilon is None:
                    writer.append((action, policy_logits[i], obs, reward, done))
                else:
                    writer.append((action, obs, reward, done))  # returns Runtime Error if a writer is closed

        progress = tf.concat([tf.constant([0.]),
                              tf.linspace(0., 1., step_counter)[:-1],
                              tf.ones(self._n_points - 2)], axis=0)
        for i, ray_writer in enumerate(ray_writers):
            steps = len(writers[i])
            # progress = tf.concat([tf.constant([0.]),
            #                       tf.linspace(0., 1., steps_per_worker_counter[i])[:-1],
            #                       tf.ones(steps - steps_per_worker_counter[i])], axis=0)

            for step in range(steps):
                action, logits, obs, reward, done = (writers[i][step][0], writers[i][step][1],
                                                     writers[i][step][2], writers[i][step][3],
                                                     writers[i][step][4])
                ray_writer.append((action, logits, obs, reward, done, rewards_saver[i], progress[step]))
                if step >= self._n_points - 1:
                    ray_writer.create_item(table=self._table_names[0], num_timesteps=self._n_points, priority=1.)

            ray_writer.close()

    def collect_once(self):
        policy = random.sample(self._policies_pool, 1)
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
            if curr_worker == self._worker_id:
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
