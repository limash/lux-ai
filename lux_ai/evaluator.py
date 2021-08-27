import random
import pickle
import time
import itertools as it
from abc import ABC

import tensorflow as tf
import numpy as np
import ray
from ray.util.queue import Empty

from tf_reinforcement_agents.abstract_agent import Agent
from tf_reinforcement_agents import models

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Evaluator(Agent, ABC):

    def __init__(self, env_name, config,
                 buffer_table_names, buffer_server_port,
                 *args, **kwargs):
        super().__init__(env_name, config,
                         buffer_table_names, buffer_server_port,
                         *args, **kwargs)

        if self._is_policy_gradient:
            # self._model = models.get_actor_critic(self._input_shape, self._n_outputs)
            self._model = models.get_actor_critic2(model_type='exp')
            self._eval_model = models.get_actor_critic2(model_type='res')
            # self._policy = self._pg_policy
        else:
            self._model = models.get_dqn(self._input_shape, self._n_outputs, is_duel=False)
            # self._policy = self._dqn_policy

        dummy_input = (tf.ones(self._input_shape[0], dtype=tf.uint8),
                       tf.ones(self._input_shape[1], dtype=tf.uint8))
        dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
        self._predict(dummy_input)
        self._eval_predict(dummy_input)

        # with open('data/data_brick.pickle', 'rb') as file:
        #     data = pickle.load(file)
        # self._model.layers[0].set_weights(data['weights'][:66])
        # self._model.layers[0].trainable = False

        try:
            with open('data/eval/data.pickle', 'rb') as file:
                data = pickle.load(file)
                print("Evaluator: using data form a data/eval/data.pickle file.")
            self._eval_model.set_weights(data['weights'])
            # self._model.set_weights(data['weights'])
        except FileNotFoundError:
            pass

    @tf.function
    def _eval_predict(self, observation):
        return self._eval_model(observation)

    def _evaluate_episode(self, epsilon):
        """
        Epsilon 0 corresponds to greedy DQN _policy,
        if epsilon is None assume policy gradient _policy
        """
        obs_records = self._eval_env.reset()
        rewards_storage = np.zeros(self._n_players)
        for step in it.count(0):
            if epsilon is None:
                actions, _ = self._policy(obs_records)
            else:
                actions = self._policy(obs_records, epsilon, info=None)
            obs_records, rewards, dones, info = self._eval_env.step(actions)
            rewards_storage += np.asarray(rewards)
            if all(dones):
                break
        return rewards_storage.mean(), step

    def _evaluate_episodes(self, num_episodes=3, epsilon=None):
        episode_rewards = 0
        steps = 0
        for _ in range(num_episodes):
            reward, step = self._evaluate_episode(epsilon)
            episode_rewards += reward
            steps += step
        return episode_rewards / num_episodes, steps / num_episodes

    def evaluate_episode(self):
        obs_records = self._eval_env.reset()
        rewards_storage = np.zeros(self._n_players)
        for step in it.count(0):
            actions = []
            obsns = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), obs_records)
            for i in range(self._n_players):
                policy_logits, _ = self._predict(obsns[i]) if i < 2 else self._eval_predict(obsns[i])
                # policy_logits, _ = self._eval_predict(obsns[i]) if i < 2 else self._predict(obsns[i])
                # if i < 2:
                #     policy_logits, _ = self._predict(obsns[i])
                # else:
                #     policy_logits, _ = self._eval_predict(obsns[i])

                action = tf.random.categorical(policy_logits, num_samples=1, dtype=tf.int32)
                actions.append(action.numpy()[0][0])

            obs_records, rewards, dones, info = self._eval_env.step(actions)
            rewards_storage += np.asarray(rewards)
            if all(dones):
                break
        # winner = rewards_storage.argmax()
        winners = np.argwhere(rewards_storage == np.amax(rewards_storage))
        return winners

    def evaluate_episodes(self):
        wins = 0
        losses = 0
        draws = 0
        for _ in range(100):
            winners = self.evaluate_episode()
            if (0 in winners or 1 in winners) and 2 not in winners and 3 not in winners:
                wins += 1
            elif (2 in winners or 3 in winners) and 0 not in winners and 1 not in winners:
                losses += 1
            else:
                draws += 1
        return wins, losses, draws

    def do_evaluate(self):
        while True:
            is_done = ray.get(self._workers_info.get_done.remote())
            if is_done:
                # print("Evaluation is done.")
                time.sleep(1)  # is needed to have time to print the last 'total wins'
                return 'Done'
            while True:
                weights, step = ray.get(self._workers_info.get_current_weights.remote())
                if weights is None:
                    time.sleep(1)
                else:
                    # print(f" Variables: {len(self._model.trainable_variables)}")
                    self._model.set_weights(weights)
                    break

            wins, losses, draws = self.evaluate_episodes()
            print(f"Evaluator: Wins: {wins}; Losses: {losses}; Draws: {draws}; Model from a step: {step}.")
