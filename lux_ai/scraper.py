import abc
import glob
import json
import random

import tensorflow as tf
import gym
import reverb

from lux_ai import tools
from lux_gym.envs.lux.action_vectors import action_vector, worker_action_mask, cart_action_mask, citytile_action_mask


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Agent(abc.ABC):

    def __init__(self, config,
                 buffer_table_names, buffer_server_port,
                 ray_queue=None, collector_id=None, workers_info=None, num_collectors=None
                 ):
        """
        Args:
            config: A configuration dictionary
            buffer_table_names: dm reverb server table names
            buffer_server_port: a port where a dm reverb server was initialized
            ray_queue: a ray interprocess queue to store neural net weights
            collector_id: to identify a current collector if there are several ones
            workers_info: a ray interprocess (remote) object to store shared information
            num_collectors: a total amount of collectors
        """
        self._n_players = 2
        self._actions_number = len(action_vector)
        self._env_name = config["environment"]

        self._feature_maps_shape = tools.get_feature_maps_shape(config["environment"])

        self._n_points = config["n_points"]

        self._table_names = buffer_table_names
        self._client = reverb.Client(f'localhost:{buffer_server_port}')

        self._ray_queue = ray_queue
        self._collector_id = collector_id
        self._workers_info = workers_info
        self._num_collectors = num_collectors

        self._files = glob.glob("./data/saved_episodes/*.json")

    def _scrape(self, data, team_name=None):
        """
        Collects trajectories from an episode to the buffer.

        A buffer contains items, each item consists of several n_points;
        One n_point contains (action, action_probs, action_mask, observation,
                              total reward, temporal_mask, progress);
        action is a response for the current observation,
        reward, done are for the current observation.
        """
        team_of_interest = -1
        if team_name:
            if data["info"]["TeamNames"][0] == team_name:
                team_of_interest = 1
            elif data["info"]["TeamNames"][1] == team_name:
                team_of_interest = 2
            else:
                return

        actions_masks = (worker_action_mask, cart_action_mask, citytile_action_mask)

        player1_data = {}
        player2_data = {}

        environment = gym.make(self._env_name, seed=data["configuration"]["seed"])
        observations, proc_obsns = environment.reset_process()
        configuration = environment.configuration
        current_game_states = environment.game_states
        width, height = current_game_states[0].map.width, current_game_states[0].map.height
        shift = int((32 - width) / 2)  # to make all feature maps 32x32

        def update_units_actions(actions, player_units_dict):
            units_with_actions = []
            for action in actions:
                action_list = action.split(" ")
                if action_list[0] not in ["r", "bw", "bc"]:
                    units_with_actions.append(action_list[1])
            for key in player_units_dict.keys():
                if key not in units_with_actions:
                    actions.append(f"m {key} c")
            return actions

        def update_cts_actions(actions, player_cts_list):
            units_with_actions = []
            for action in actions:
                action_list = action.split(" ")
                if action_list[0] in ["r", "bw", "bc"]:
                    x, y = int(action_list[1]), int(action_list[2])
                    units_with_actions.append([x, y])
            for ct in player_cts_list:
                if ct not in units_with_actions:
                    actions.append(f"idle {ct[0]} {ct[1]}")
            return actions

        def get_actions_dict(player_actions, player_units_dict_active, player_units_dict_all):
            actions_dict = {"workers": {}, "carts": {}, "city_tiles": {}}
            for action in player_actions:
                action_list = action.split(" ")
                # units
                if action_list[0] == "m":  # "m {id} {direction}"
                    unit_name = action_list[1]  # "{id}"
                    direction = action_list[2]
                    try:
                        unit_type = "w" if player_units_dict_active[unit_name].is_worker() else "c"
                    except KeyError:  # it occurs when there is not valid action proposed
                        continue
                    action_vector_name = f"{unit_type}_m{direction}"  # "{unit_type}_m{direction}"
                    if unit_type == "w":
                        actions_dict["workers"][unit_name] = action_vector[action_vector_name]
                    else:
                        actions_dict["carts"][unit_name] = action_vector[action_vector_name]
                elif action_list[0] == "t":  # "t {id} {dest_id} {resourceType} {amount}"
                    unit_name = action_list[1]
                    dest_name = action_list[2]
                    resourceType = action_list[3]
                    try:
                        unit_type = "w" if player_units_dict_active[unit_name].is_worker() else "c"
                    except KeyError:  # it occurs when there is not valid action proposed
                        continue
                    direction = player_units_dict_active[unit_name].pos.direction_to(player_units_dict_all[
                                                                                         dest_name].pos)
                    action_vector_name = f"{unit_type}_t{direction}{resourceType}"
                    if unit_type == "w":
                        actions_dict["workers"][unit_name] = action_vector[action_vector_name]
                    else:
                        actions_dict["carts"][unit_name] = action_vector[action_vector_name]
                elif action_list[0] == "bcity":  # "bcity {id}"
                    unit_name = action_list[1]
                    action_vector_name = "w_build"
                    actions_dict["workers"][unit_name] = action_vector[action_vector_name]
                elif action_list[0] == "p":  # "p {id}"
                    unit_name = action_list[1]
                    action_vector_name = "w_pillage"
                    actions_dict["workers"][unit_name] = action_vector[action_vector_name]
                # city tiles
                elif action_list[0] == "r":  # "r {pos.x} {pos.y}"
                    x, y = int(action_list[1]), int(action_list[2])
                    unit_name = f"ct_{y + shift}_{x + shift}"
                    action_vector_name = "ct_research"
                    actions_dict["city_tiles"][unit_name] = action_vector[action_vector_name]
                elif action_list[0] == "bw":  # "bw {pos.x} {pos.y}"
                    x, y = int(action_list[1]), int(action_list[2])
                    unit_name = f"ct_{y + shift}_{x + shift}"
                    action_vector_name = "ct_build_worker"
                    actions_dict["city_tiles"][unit_name] = action_vector[action_vector_name]
                elif action_list[0] == "bc":  # "bc {pos.x} {pos.y}"
                    x, y = int(action_list[1]), int(action_list[2])
                    unit_name = f"ct_{y + shift}_{x + shift}"
                    action_vector_name = "ct_build_cart"
                    actions_dict["city_tiles"][unit_name] = action_vector[action_vector_name]
                elif action_list[0] == "idle":  # "idle {pos.x} {pos.y}"
                    x, y = int(action_list[1]), int(action_list[2])
                    unit_name = f"ct_{y + shift}_{x + shift}"
                    action_vector_name = "ct_idle"
                    actions_dict["city_tiles"][unit_name] = action_vector[action_vector_name]
                else:
                    raise ValueError
            return actions_dict

        def add_point(player_data, actions_dict, actions_probs, proc_obs, current_step):
            for i, (acts, acts_prob, obs) in enumerate(zip(actions_dict.values(),
                                                           actions_probs.values(),
                                                           proc_obs.values())):
                acts = dict(sorted(acts.items()))
                acts_prob = dict(sorted(acts_prob.items()))
                obs = dict(sorted(obs.items()))
                for (k1, action), (k2, action_probs), (k3, observation) in zip(acts.items(),
                                                                               acts_prob.items(),
                                                                               obs.items()):
                    assert k1 == k2 == k3
                    point_value = [action, action_probs, actions_masks[i], observation]
                    if k1 in player_data.keys():
                        player_data[k1].append(point_value, current_step)
                    else:
                        player_data[k1] = tools.DataValue()
                        player_data[k1].append(point_value, current_step)
            return player_data

        step = 0
        for step in range(0, configuration.episodeSteps):
            assert observations[0]["updates"] == observations[1]["updates"] == data["steps"][step][0]["observation"][
                "updates"]
            # get units to know their types etc.
            player1 = current_game_states[0].players[observations[0].player]
            player1_units_dict_active = {}
            player1_units_dict_all = {}
            player2 = current_game_states[1].players[(observations[0].player + 1) % 2]
            player2_units_dict_active = {}
            player2_units_dict_all = {}
            for unit in player1.units:
                player1_units_dict_all[unit.id] = unit
                if unit.can_act():
                    player1_units_dict_active[unit.id] = unit
            for unit in player2.units:
                player2_units_dict_all[unit.id] = unit
                if unit.can_act():
                    player2_units_dict_active[unit.id] = unit
            # get citytiles
            player1_ct_list_active = []
            for city in player1.cities.values():
                for citytile in city.citytiles:
                    if citytile.cooldown < 1:
                        x_coord, y_coord = citytile.pos.x, citytile.pos.y
                        player1_ct_list_active.append([x_coord, y_coord])
            player2_ct_list_active = []
            for city in player2.cities.values():
                for citytile in city.citytiles:
                    if citytile.cooldown < 1:
                        x_coord, y_coord = citytile.pos.x, citytile.pos.y
                        player2_ct_list_active.append([x_coord, y_coord])
            # get actions from a record, action for the current obs is in the next step of data
            actions_1 = data["steps"][step + 1][0]["action"]
            actions_2 = data["steps"][step + 1][1]["action"]
            # if no action and unit can act, add "m {id} c"
            actions_1_vec = actions_1.copy()
            actions_2_vec = actions_2.copy()
            actions_1_vec = update_units_actions(actions_1_vec, player1_units_dict_active)
            actions_2_vec = update_units_actions(actions_2_vec, player2_units_dict_active)
            # in no action and ct can act, add "idle {x} {y}"
            actions_1_vec = update_cts_actions(actions_1_vec, player1_ct_list_active)
            actions_2_vec = update_cts_actions(actions_2_vec, player2_ct_list_active)

            # get actions vector representation
            actions_1_dict = get_actions_dict(actions_1_vec, player1_units_dict_active, player1_units_dict_all)
            actions_2_dict = get_actions_dict(actions_2_vec, player2_units_dict_active, player2_units_dict_all)

            # probs are similar to actions
            player1_data = add_point(player1_data, actions_1_dict, actions_1_dict, proc_obsns[0], step)
            player2_data = add_point(player2_data, actions_2_dict, actions_2_dict, proc_obsns[1], step)

            dones, observations, proc_obsns = environment.step_process((actions_1, actions_2))
            current_game_states = environment.game_states

            if any(dones):
                break

        count_team_1 = 0
        for value in list(player1_data.values()):
            count_team_1 += len(value.data)
        count_team_2 = 0
        for value in list(player2_data.values()):
            count_team_2 += len(value.data)
        print(f"Team 1 count: {count_team_1}; Team 2 count: {count_team_2}; Team to add: {team_of_interest}")

        progress = tf.linspace(0., 1., step + 2)[:-1]
        progress = tf.cast(progress, dtype=tf.float16)

        reward1, reward2 = data["rewards"][0], data["rewards"][1]
        if reward1 > reward2:
            final_reward_1 = tf.constant(1, dtype=tf.float16)
            final_reward_2 = tf.constant(-1, dtype=tf.float16)
        elif reward1 < reward2:
            final_reward_2 = tf.constant(1, dtype=tf.float16)
            final_reward_1 = tf.constant(-1, dtype=tf.float16)
        else:
            final_reward_1 = final_reward_2 = tf.constant(0, dtype=tf.float16)

        obs_zeros = tf.zeros(self._feature_maps_shape, dtype=tf.float16)
        act_zeros = tf.zeros(self._actions_number, dtype=tf.float16)
        act_ones = tf.ones(self._actions_number, dtype=tf.float16)
        act_probs_uni = tf.ones(self._actions_number, dtype=tf.float16) * 1/self._actions_number

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

        if team_of_interest == -1:
            send_data(player1_data, final_reward_1)
            send_data(player2_data, final_reward_2)
        elif team_of_interest == 1:
            send_data(player1_data, final_reward_1)
        elif team_of_interest == 2:
            send_data(player2_data, final_reward_2)

    def scrape_once(self):
        file_name = random.sample(self._files, 1)[0]
        with open(file_name, "r") as read_file:
            data = json.load(read_file)
        self._scrape(data)

    def scrape_all(self):
        for file_name in self._files:
            with open(file_name, "r") as read_file:
                print(f"File is {file_name}")
                data = json.load(read_file)
            self._scrape(data, team_name="Looking for Halite")
        try:
            checkpoint = self._client.checkpoint()
        except RuntimeError as err:
            print(err)
            checkpoint = err
