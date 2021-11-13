import random

import numpy as np
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

AUTO = tf.data.experimental.AUTOTUNE


# Three data types can be stored in TFRecords: bytestrings, integers and floats
# They are always stored as lists, a single data element is a list of size 1
def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def _float_feature(list_of_floats):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def _int_feature(list_of_ints):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def to_tfrecord(reward, action_probs, observation):
    feature = {
        "observation": _bytestring_feature([observation]),
        "action_probs_1": _bytestring_feature([action_probs[0]]),
        "action_probs_2": _bytestring_feature([action_probs[1]]),
        "action_probs_3": _bytestring_feature([action_probs[2]]),
        "reward": _float_feature([reward]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def to_tfrecord_for_rl(actions_numbers, actions_probs, observations, rewards, masks, progress_array, final_idx):
    feature = {
        "actions_numbers": _bytestring_feature([actions_numbers]),
        "actions_probs_1": _bytestring_feature([actions_probs[0]]),
        "actions_probs_2": _bytestring_feature([actions_probs[1]]),
        "actions_probs_3": _bytestring_feature([actions_probs[2]]),
        "observations": _bytestring_feature([observations]),
        "rewards": _bytestring_feature([rewards]),
        "masks": _bytestring_feature([masks]),
        "progress_array": _bytestring_feature([progress_array]),
        "final_idx": _int_feature([final_idx]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def to_tfrecord_for_rl_pg(action_numbers, action_probs, observation, reward, progress_value):
    feature = {
        "action_numbers": _bytestring_feature([action_numbers]),
        "action_probs_1": _bytestring_feature([action_probs[0]]),
        "action_probs_2": _bytestring_feature([action_probs[1]]),
        "action_probs_3": _bytestring_feature([action_probs[2]]),
        "observation": _bytestring_feature([observation]),
        "reward": _float_feature([reward]),
        "progress_value": _float_feature([progress_value]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(ds, record_number, record_name, is_for_rl, save_path=None, collector_n=None,
                   is_pg_rl=False):
    if is_pg_rl:
        if save_path is None:
            save_path = "data/tfrecords/rl/storage/"
        out_file = None
        for n, (action_numbers, action_probs, observation, reward, progress_value) in enumerate(ds):
            if out_file is None or n % 3000 == 0:
                if out_file is not None:
                    out_file.close()
                if collector_n is not None:
                    filename = f"{save_path}{collector_n}_{record_name}_{n}.tfrec"
                else:
                    filename = f"{save_path}{record_name}_{n}.tfrec"
                out_file = tf.io.TFRecordWriter(filename)
            s_action_numbers = tf.io.serialize_tensor(action_numbers)
            serial_action_probs = [tf.io.serialize_tensor(item).numpy() for item in action_probs]
            serial_observation = tf.io.serialize_tensor(tf.io.serialize_sparse(observation))
            example = to_tfrecord_for_rl_pg(
                s_action_numbers.numpy(),
                serial_action_probs,
                serial_observation.numpy(),
                reward.numpy().astype(np.float32),
                progress_value.numpy().astype(np.float32),
            )
            out_file.write(example.SerializeToString())
        out_file.close()
        print(f"Wrote group #{record_number} {record_name} tfrec files containing {n} records")
    elif is_for_rl:
        if save_path is None:
            save_path = "data/tfrecords/rl/storage/"
        out_file = None
        for n, (actions_numbers, actions_probs, observations, rewards,
                masks, progress_array, final_idx) in enumerate(ds):
            if out_file is None or n % 10 == 0:
                if out_file is not None:
                    out_file.close()
                if collector_n is not None:
                    filename = f"{save_path}{collector_n}_{record_name}_{n}.tfrec"
                else:
                    filename = f"{save_path}{record_name}_{n}.tfrec"
                out_file = tf.io.TFRecordWriter(filename)
            s_actions_numbers = tf.io.serialize_tensor(actions_numbers)
            serial_action_probs = [tf.io.serialize_tensor(item).numpy() for item in actions_probs]
            s_observations = tf.io.serialize_tensor(tf.io.serialize_sparse(observations))
            s_rewards = tf.io.serialize_tensor(rewards)
            s_masks = tf.io.serialize_tensor(masks)
            s_progress_array = tf.io.serialize_tensor(progress_array)
            example = to_tfrecord_for_rl(
                s_actions_numbers.numpy(),
                serial_action_probs,
                s_observations.numpy(),
                s_rewards.numpy(),
                s_masks.numpy(),
                s_progress_array.numpy(),
                final_idx.numpy().astype(np.int64),
            )
            out_file.write(example.SerializeToString())
        out_file.close()
        print(f"Wrote group #{record_number} {record_name} tfrec files containing {n} records")
    else:
        if save_path is None:
            save_path = "data/tfrecords/imitator/train/"
        out_file = None
        # with tf.io.TFRecordWriter(filename) as out_file:
        for n, (observation, action_probs, reward) in enumerate(ds):
            if out_file is None or n % 3000 == 0:
                if out_file is not None:
                    out_file.close()
                if collector_n is not None:
                    filename = f"{save_path}{collector_n}_{record_name}_{n}.tfrec"
                else:
                    filename = f"{save_path}{record_name}_{n}.tfrec"
                out_file = tf.io.TFRecordWriter(filename)
            serial_action_probs = [tf.io.serialize_tensor(item).numpy() for item in action_probs]
            serial_observation = tf.io.serialize_tensor(tf.io.serialize_sparse(observation))
            example = to_tfrecord(reward.numpy().astype(np.float32),
                                  serial_action_probs,
                                  serial_observation.numpy())
            out_file.write(example.SerializeToString())
        out_file.close()
        print(f"Wrote group #{record_number} {record_name} tfrec files containing {n} records")


def record(player1_data, player2_data, final_reward_1, final_reward_2,
           feature_maps_shape, actions_shape, record_number, record_name,
           progress=None, is_for_rl=False, save_path=None, collector_n=None, is_pg_rl=False):
    def data_gen_all():
        for j, player_data in enumerate((player1_data, player2_data)):
            if player_data is None:
                continue
            final_reward = final_reward_1 if j == 0 else final_reward_2
            for key, unit in player_data.items():
                unit_type = key.split("_")[0]
                if unit_type != "u":
                    continue
                actions = unit.actions
                counters = np.zeros_like(actions)  # for debug, shows points with actions yielded
                for point in unit.data:
                    # point [action, action_probs, observation]
                    action = point[0]
                    # if action[4] == 1. and random.random() > 0.05:
                    #     continue
                    # store observation, action_probs, reward
                    observation = tf.sparse.from_dense(tf.constant(point[2], dtype=tf.float16))
                    actions_probs = tuple([tf.constant(item, dtype=tf.float16) for item in point[1]])
                    reward = tf.constant(final_reward, dtype=tf.float16)
                    yield observation, actions_probs, reward
                    # yield point[2], point[1], final_reward
                    counters += action[0]

    episode_length = 360
    trajectory_steps = 40
    total_len = episode_length + trajectory_steps

    def data_gen_all_for_rl():
        for j, player_data in enumerate((player1_data, player2_data)):
            if player_data is None:
                continue
            final_reward = final_reward_1 if j == 0 else final_reward_2
            for key, unit in player_data.items():
                unit_type = key.split("_")[0]
                if unit_type != "u":
                    continue
                actions_numbers = -np.ones([total_len] + [len(unit.data[0][1])])
                actions_probs = [np.zeros([total_len] + list(item.shape)) for item in unit.data[0][1]]
                observations = np.zeros([total_len] + list(unit.data[0][2].shape))
                rewards = np.zeros([total_len])
                masks = np.zeros([total_len])
                progress_array = np.zeros([total_len])
                i = 0
                for i, point in enumerate(unit.data):
                    for n, item in enumerate(point[0]):
                        non_zeros = np.count_nonzero(item)
                        if non_zeros:
                            actions_numbers[i][n] = np.argmax(item)
                    for n, prob_item in enumerate(point[1]):
                        actions_probs[n][i] = prob_item
                    observations[i] = point[2]
                    rewards[i] = final_reward
                    masks[i] = 1
                    progress_array[i] = progress[unit.step[i]].numpy()
                # cast to tf tensors
                actions_numbers = tf.constant(actions_numbers, dtype=tf.int16)
                actions_probs = tuple([tf.constant(item, dtype=tf.float16) for item in actions_probs])
                observations = tf.sparse.from_dense(tf.constant(observations, dtype=tf.float16))
                rewards = tf.constant(rewards, dtype=tf.float16)
                masks = tf.constant(masks, dtype=tf.int16)
                progress_array = tf.constant(progress_array, dtype=tf.float16)
                final_idx = tf.constant(i, dtype=tf.int16)

                yield actions_numbers, actions_probs, observations, rewards, masks, progress_array, final_idx

    def data_gen_soft():
        for j, player_data in enumerate((player1_data, player2_data)):
            if player_data is None:
                continue
            final_reward = final_reward_1 if j == 0 else final_reward_2
            for key, unit in player_data.items():
                unit_type = key.split("_")[0]
                if unit_type != "u":
                    continue
                # median = np.median(unit.actions[np.nonzero(unit.actions)])
                mean = np.mean(unit.actions[np.nonzero(unit.actions)])
                actions = unit.actions
                # multipliers = np.divide(median, actions, out=np.zeros_like(actions), where=actions != 0)
                multipliers = np.divide(mean, actions, out=np.zeros_like(actions), where=actions != 0)
                final_idx = len(unit.data)
                counters = np.zeros_like(actions)  # for debug, shows points with actions yielded
                if np.nonzero(unit.actions)[0].shape[0] == 1:
                    # if only one possible action in unit episode trajectory, add only the last one (issue?)
                    idx = final_idx - 1
                else:
                    idx = 0
                while True:
                    # point [action, action_probs, observation]
                    point = unit.data[idx]
                    action = point[0]
                    multiplier = multipliers[np.nonzero(action)][0]

                    if multiplier > 1.01 and random.random() > 1 / multiplier:
                        # repeat point
                        idx -= 1
                    elif multiplier < 0.99 and random.random() > multiplier:
                        # skip point
                        idx += 1
                        if idx == final_idx:
                            break
                        continue

                    # store observation, action_probs, reward
                    observation = tf.sparse.from_dense(tf.constant(point[2], dtype=tf.float16))
                    actions_probs = tf.constant(point[1], dtype=tf.float16)
                    reward = tf.constant(final_reward, dtype=tf.float16)
                    yield observation, actions_probs, reward
                    counters += action

                    idx += 1
                    if idx == final_idx:
                        break

    def data_gen_soft_for_pg_no_transfers():
        for j, player_data in enumerate((player1_data, player2_data)):
            if player_data is None:
                continue
            final_reward = final_reward_1 if j == 0 else final_reward_2
            for key, unit in player_data.items():
                unit_type = key.split("_")[0]
                if unit_type != "u":
                    continue
                actions = unit.actions
                movements_average = actions[0] / 4 if actions[0] > 0 else 1.
                idle_prob = movements_average / actions[2] if actions[2] > 0 else 1.
                build_multiplier = movements_average / actions[3] if actions[3] > 0 else 1.
                counters = np.zeros_like(actions)  # for debug, shows points with actions yielded
                build_repeat_counter = 0
                final_idx = len(unit.data)
                idx = 0
                while True:
                    # point [action_vectors, action_vectors_probs, observation]
                    point = unit.data[idx]
                    progress_value = progress[unit.step[idx]].numpy()

                    action_vectors = point[0]
                    # choice
                    general_actions = action_vectors[0]
                    act_index = np.argmax(general_actions)
                    if act_index == 1:  # transfer to skip
                        idx += 1
                        if idx == final_idx:
                            break
                        continue
                    elif act_index == 2:  # idle
                        if idle_prob > 1:
                            current_idle_prob = 1.
                        else:
                            current_idle_prob = max(0.15, (2 * (final_idx - idx) / final_idx) * idle_prob)
                        if random.random() > current_idle_prob:
                            idx += 1
                            if idx == final_idx:
                                break
                            continue
                    elif act_index == 3:  # bcity
                        if random.random() > 1 / build_multiplier:
                            idx -= 1
                            build_repeat_counter += 1
                            if build_repeat_counter > 6:  # do not repeat too much
                                idx += 1
                                build_repeat_counter = 0

                    #
                    actions_numbers = -np.ones(len(action_vectors))
                    for n, item in enumerate(action_vectors):
                        non_zeros = np.count_nonzero(item)
                        if non_zeros:
                            actions_numbers[n] = np.argmax(item)
                    actions_numbers = tf.constant(actions_numbers, dtype=tf.int16)
                    #
                    actions_probs = tuple([tf.constant(item, dtype=tf.float16) for item in point[1]])
                    #
                    observation = tf.sparse.from_dense(tf.constant(point[2], dtype=tf.float16))
                    #
                    reward = tf.constant(final_reward, dtype=tf.float16)
                    #
                    progress_value = tf.constant(progress_value, dtype=tf.float16)
                    #
                    yield actions_numbers, actions_probs, observation, reward, progress_value
                    counters += general_actions

                    idx += 1
                    if idx == final_idx:
                        break

    # result = []
    # generator = data_gen_soft_for_pg_no_transfers()
    # while len(result) < 10000:
    #     x = next(generator)
    #     result.append(x)

    if is_pg_rl:
        dataset = tf.data.Dataset.from_generator(
            data_gen_soft_for_pg_no_transfers,
            output_signature=(
                tf.TensorSpec(shape=len(actions_shape), dtype=tf.int16),
                tuple([tf.TensorSpec(shape=actions_number, dtype=tf.float16) for actions_number in actions_shape]),
                tf.SparseTensorSpec(shape=feature_maps_shape, dtype=tf.float16),
                tf.TensorSpec(shape=(), dtype=tf.float16),
                tf.TensorSpec(shape=(), dtype=tf.float16),
            ))
    elif is_for_rl:
        dataset = tf.data.Dataset.from_generator(
            data_gen_all_for_rl,
            output_signature=(
                tf.TensorSpec(shape=[total_len, len(actions_shape)], dtype=tf.int16),
                tuple([tf.TensorSpec(shape=[total_len] + list(actions_number), dtype=tf.float16)
                       for actions_number in actions_shape]),
                tf.SparseTensorSpec(shape=[total_len] + list(feature_maps_shape), dtype=tf.float16),
                tf.TensorSpec(shape=total_len, dtype=tf.float16),
                tf.TensorSpec(shape=total_len, dtype=tf.int16),
                tf.TensorSpec(shape=total_len, dtype=tf.float16),
                tf.TensorSpec(shape=(), dtype=tf.int16),
            ))
    else:
        dataset = tf.data.Dataset.from_generator(
            data_gen_all,
            output_signature=(
                tf.SparseTensorSpec(shape=feature_maps_shape, dtype=tf.float16),
                tuple([tf.TensorSpec(shape=actions_number, dtype=tf.float16) for actions_number in actions_shape]),
                tf.TensorSpec(shape=(), dtype=tf.float16),
            ))

    # foo = list(dataset.take(1))
    write_tfrecord(dataset, record_number, record_name, is_for_rl, save_path, collector_n, is_pg_rl)


def up_down(obs, probs):
    obs = tf.reverse(obs, [0])
    moves = probs[1]
    probs[1] = tf.stack([moves[2], moves[1], moves[0], moves[3]], axis=0)
    return obs, probs


def left_right(obs, probs):
    obs = tf.reverse(obs, [1])
    moves = probs[1]
    probs[1] = tf.stack([moves[0], moves[3], moves[2], moves[1]], axis=0)
    return obs, probs


def do_nothing(obs, probs):
    return obs, probs


def random_reverse(observations, inputs):
    act_probs, dir_probs, res_probs, reward = inputs
    probs = [act_probs, dir_probs, res_probs]

    # trigger = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    trigger = np.random.choice(np.array([0, 1, 2, 3]))

    # observations, probs = tf.cond(tf.equal(trigger, tf.constant(1, dtype=tf.int32)),
    #                               lambda: up_down(observations, probs),
    #                               lambda: do_nothing(observations, probs))
    if trigger == 1:
        observations, probs = up_down(observations, probs)
    elif trigger == 2:
        observations, probs = left_right(observations, probs)
    elif trigger == 3:
        observations, probs = up_down(observations, probs)
        observations, probs = left_right(observations, probs)

    act_probs, dir_probs, res_probs = probs
    return observations, (act_probs, dir_probs, res_probs, reward)


def random_reverse_pg(act_numbers, act_probs, dir_probs, res_probs, observations, reward, progress):
    probs = [act_probs, dir_probs, res_probs]

    new_act_numbers = act_numbers
    trigger = np.random.choice(np.array([0, 1, 2, 3]))
    if trigger == 1:
        observations, probs = up_down(observations, probs)
        if act_numbers[1] == 0:
            new_act_numbers = tf.stack([act_numbers[0], tf.constant(2, dtype=tf.int32), act_numbers[2]], axis=0)
        elif act_numbers[1] == 2:
            new_act_numbers = tf.stack([act_numbers[0], tf.constant(0, dtype=tf.int32), act_numbers[2]], axis=0)
    elif trigger == 2:
        observations, probs = left_right(observations, probs)
        if act_numbers[1] == 1:
            new_act_numbers = tf.stack([act_numbers[0], tf.constant(3, dtype=tf.int32), act_numbers[2]], axis=0)
        elif act_numbers[1] == 3:
            new_act_numbers = tf.stack([act_numbers[0], tf.constant(1, dtype=tf.int32), act_numbers[2]], axis=0)
    elif trigger == 3:
        observations, probs = up_down(observations, probs)
        observations, probs = left_right(observations, probs)
        if act_numbers[1] == 0:
            new_act_numbers = tf.stack([act_numbers[0], tf.constant(2, dtype=tf.int32), act_numbers[2]], axis=0)
        elif act_numbers[1] == 2:
            new_act_numbers = tf.stack([act_numbers[0], tf.constant(0, dtype=tf.int32), act_numbers[2]], axis=0)
        elif act_numbers[1] == 1:
            new_act_numbers = tf.stack([act_numbers[0], tf.constant(3, dtype=tf.int32), act_numbers[2]], axis=0)
        elif act_numbers[1] == 3:
            new_act_numbers = tf.stack([act_numbers[0], tf.constant(1, dtype=tf.int32), act_numbers[2]], axis=0)

    act_probs, dir_probs, res_probs = probs
    return new_act_numbers, act_probs, dir_probs, res_probs, observations, reward, progress


def split_movement_actions(observation, inputs):
    action_probs_1, action_probs_2, action_probs_3, reward = inputs
    zeros = tf.constant([0, 0, 0, 0], dtype=tf.float32)
    if action_probs_1[0] == 1:  # movement action
        # action type, movement direction, transfer direction, resource to transfer, reward
        return observation, (action_probs_1, action_probs_2, zeros, action_probs_3, reward)
    elif action_probs_1[1] == 1:  # transfer action
        return observation, (action_probs_1, zeros, action_probs_2, action_probs_3, reward)
    else:
        return observation, (action_probs_1, zeros, zeros, action_probs_3, reward)


def merge_actions(observation, inputs):
    act_probs, dir_probs, res_probs, reward = inputs
    movements = dir_probs * act_probs[0]
    idle = act_probs[1:2] + act_probs[2:3]  # transfer + idle
    bcity = act_probs[3:]
    row_probs = tf.concat([movements, idle, bcity], axis=0)
    row_logs = tf.math.log(row_probs)  # it produces infs, but softmax seems to be fine with it
    new_probs = tf.nn.softmax(row_logs)  # normalize action probs
    if tf.reduce_any(tf.math.is_nan(new_probs)):
        new_probs = row_probs
    return observation, (new_probs, reward)


def merge_actions_pg(act_numbers, act_probs, dir_probs, res_probs, observations, reward, progress):
    movements = dir_probs * act_probs[0]
    idle = act_probs[1:2] + act_probs[2:3]  # transfer + idle
    bcity = act_probs[3:]
    row_probs = tf.concat([movements, idle, bcity], axis=0)
    row_logs = tf.math.log(row_probs)  # it produces infs, but softmax seems to be fine with it
    new_probs = tf.nn.softmax(row_logs)  # normalize action probs
    if tf.reduce_any(tf.math.is_nan(new_probs)):
        new_probs = row_probs
    if act_numbers[0] == 0:
        if act_numbers[1] == 0:
            act_number = 0
        elif act_numbers[1] == 1:
            act_number = 1
        elif act_numbers[1] == 2:
            act_number = 2
        elif act_numbers[1] == 3:
            act_number = 3
        else:
            act_number = -1
    elif act_numbers[0] == 2:
        act_number = 4
    elif act_numbers[0] == 3:
        act_number = 5
    else:
        act_number = -1
    return act_number, new_probs, observations, reward, progress


def merge_actions_amplify(observation, inputs):
    act_probs, dir_probs, res_probs, reward = inputs
    movements = dir_probs * act_probs[0]
    idle = act_probs[1:2] + act_probs[2:3]  # transfer + idle
    bcity = act_probs[3:]
    row_probs = tf.concat([movements, idle, bcity], axis=0)
    row_logs = tf.math.log(row_probs)  # it produces infs, but softmax seems to be fine with it
    new_probs = tf.nn.softmax(row_logs*2)  # normalize and amplify action probs
    if tf.reduce_any(tf.math.is_nan(new_probs)):
        new_probs = row_probs
    return observation, (new_probs, reward)


def merge_actions_rl(act_numbers, act_probs, dir_probs, res_probs, observation, reward, mask, progress):
    # steps = act_probs.shape[0]
    steps = 40
    ta = tf.TensorArray(dtype=tf.float32, size=steps, dynamic_size=False)
    for i in tf.range(steps):
        movements = dir_probs[i] * act_probs[i][0]
        idle = act_probs[i][1:2] + act_probs[i][2:3]  # transfer + idle
        bcity = act_probs[i][3:]
        row_probs = tf.concat([movements, idle, bcity], axis=0)
        row_logs = tf.math.log(row_probs)  # it produces infs, but softmax seems to be fine with it
        row = tf.nn.softmax(row_logs)  # normalize action probs
        if tf.reduce_any(tf.math.is_nan(row)):
            row = row_probs
        ta = ta.write(i, row)
    new_probs = ta.stack()
    act_numbers = tf.argmax(new_probs, axis=1)
    return act_numbers, new_probs, observation, reward, mask, progress


def read_records_for_imitator(feature_maps_shape, actions_shape, model_name, path,
                              filenames=None, amplify_probs=False):
    # read from TFRecords. For optimal performance, read from multiple
    # TFRecord files at once and set the option experimental_deterministic = False
    # to allow order-altering optimizations.

    def read_tfrecord(example):
        features = {
            "observation": tf.io.FixedLenFeature([], tf.string),
            "action_probs_1": tf.io.FixedLenFeature([], tf.string),
            "action_probs_2": tf.io.FixedLenFeature([], tf.string),
            "action_probs_3": tf.io.FixedLenFeature([], tf.string),
            "reward": tf.io.FixedLenFeature([], tf.float32),
        }
        # decode the TFRecord
        example = tf.io.parse_single_example(example, features)

        # observation = tf.io.parse_tensor(example["observation"], tf.float16)
        observation = tf.io.parse_tensor(example["observation"], tf.string)
        observation = tf.expand_dims(observation, axis=0)
        observation = tf.io.deserialize_many_sparse(observation, dtype=tf.float16)
        observation = tf.sparse.to_dense(observation)
        observation = tf.squeeze(observation)
        observation = tf.cast(observation, dtype=tf.float32)
        observation.set_shape(feature_maps_shape)
        # action_mask = tf.io.parse_tensor(example["action_mask"], tf.float16)
        # action_mask.set_shape(actions_shape)
        action_probs_1 = tf.io.parse_tensor(example["action_probs_1"], tf.float16)
        action_probs_1 = tf.cast(action_probs_1, dtype=tf.float32)
        action_probs_1.set_shape(actions_shape[0][0])
        action_probs_2 = tf.io.parse_tensor(example["action_probs_2"], tf.float16)
        action_probs_2 = tf.cast(action_probs_2, dtype=tf.float32)
        action_probs_2.set_shape(actions_shape[1][0])
        action_probs_3 = tf.io.parse_tensor(example["action_probs_3"], tf.float16)
        action_probs_3 = tf.cast(action_probs_3, dtype=tf.float32)
        action_probs_3.set_shape(actions_shape[2][0])
        # action_probs = (action_probs_1, action_probs_2, action_probs_3)
        reward = example["reward"]
        reward.set_shape(())
        reward = tf.cast(reward, dtype=tf.float32)

        return observation, (action_probs_1, action_probs_2, action_probs_3, reward)

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    if filenames is None:
        filenames = tf.io.gfile.glob(path + "*.tfrec")

    # test_dataset = tf.data.Dataset.list_files(filenames)
    # test_dataset = test_dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
    #                                        cycle_length=5,
    #                                        num_parallel_calls=AUTO,
    #                                        )
    # count = 0
    # for item in test_dataset:
    #     foo = read_tfrecord(item)
    #     foo = random_reverse(*foo)
    #     foo = merge_actions_amplify(*foo)
    #     # foo = split_movement_actions(*foo)
    #     count += 1

    # filenames_ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    filenames_ds = tf.data.Dataset.list_files(filenames)
    # filenames_ds = filenames_ds.with_options(option_no_order)
    ds = filenames_ds.interleave(lambda x: tf.data.TFRecordDataset(x),
                                 cycle_length=5,
                                 num_parallel_calls=AUTO
                                 )
    ds = ds.map(read_tfrecord, num_parallel_calls=AUTO)
    ds = ds.map(random_reverse, num_parallel_calls=AUTO)
    if model_name == "actor_critic_residual_shrub":
        ds = ds.map(split_movement_actions, num_parallel_calls=AUTO)
    elif model_name == "actor_critic_residual_six_actions":
        if amplify_probs:
            ds = ds.map(merge_actions_amplify, num_parallel_calls=AUTO)
        else:
            ds = ds.map(merge_actions, num_parallel_calls=AUTO)
    else:
        raise NotImplementedError
    ds = ds.shuffle(10000)
    return ds


def read_records_for_rl(feature_maps_shape, actions_shape, trajectory_steps, model_name, path):
    # read from TFRecords. For optimal performance, read from multiple
    # TFRecord files at once and set the option experimental_deterministic = False
    # to allow order-altering optimizations.
    episode_length = 360
    total_len = episode_length + trajectory_steps

    def read_tfrecord(example):
        features = {
            "actions_numbers": tf.io.FixedLenFeature([], tf.string),
            "actions_probs_1": tf.io.FixedLenFeature([], tf.string),
            "actions_probs_2": tf.io.FixedLenFeature([], tf.string),
            "actions_probs_3": tf.io.FixedLenFeature([], tf.string),
            "observations": tf.io.FixedLenFeature([], tf.string),
            "rewards": tf.io.FixedLenFeature([], tf.string),
            "masks": tf.io.FixedLenFeature([], tf.string),
            "progress_array": tf.io.FixedLenFeature([], tf.string),
            "final_idx": tf.io.FixedLenFeature([], tf.int64),
        }
        # decode the TFRecord
        example = tf.io.parse_single_example(example, features)

        actions_numbers = tf.io.parse_tensor(example["actions_numbers"], tf.int16)
        actions_numbers.set_shape([total_len, len(actions_shape)])

        actions_probs_1 = tf.io.parse_tensor(example["actions_probs_1"], tf.float16)
        actions_probs_1.set_shape([total_len] + list(actions_shape[0]))
        actions_probs_2 = tf.io.parse_tensor(example["actions_probs_2"], tf.float16)
        actions_probs_2.set_shape([total_len] + list(actions_shape[1]))
        actions_probs_3 = tf.io.parse_tensor(example["actions_probs_3"], tf.float16)
        actions_probs_3.set_shape([total_len] + list(actions_shape[2]))

        observations = tf.io.parse_tensor(example["observations"], tf.string)
        observations = tf.expand_dims(observations, axis=0)
        observations = tf.io.deserialize_many_sparse(observations, dtype=tf.float16)
        observations = tf.sparse.to_dense(observations)
        observations = tf.squeeze(observations)
        observations.set_shape([total_len] + list(feature_maps_shape))

        rewards = tf.io.parse_tensor(example["rewards"], tf.float16)
        rewards.set_shape(total_len)

        masks = tf.io.parse_tensor(example["masks"], tf.int16)
        masks.set_shape(total_len)

        progress_array = tf.io.parse_tensor(example["progress_array"], tf.float16)
        progress_array.set_shape(total_len)

        final_idx = example["final_idx"]
        final_idx.set_shape(())
        start_idx = tf.random.uniform(shape=(), minval=0, maxval=final_idx+1, dtype=tf.int64)

        return tf.cast(actions_numbers[start_idx: start_idx + trajectory_steps, :], dtype=tf.int32), \
               tf.cast(actions_probs_1[start_idx: start_idx + trajectory_steps, :], dtype=tf.float32), \
               tf.cast(actions_probs_2[start_idx: start_idx + trajectory_steps, :], dtype=tf.float32), \
               tf.cast(actions_probs_3[start_idx: start_idx + trajectory_steps, :], dtype=tf.float32), \
               tf.cast(observations[start_idx: start_idx + trajectory_steps, :, :, :], dtype=tf.float32), \
               tf.cast(rewards[start_idx: start_idx + trajectory_steps], dtype=tf.float32), \
               tf.cast(masks[start_idx: start_idx + trajectory_steps], dtype=tf.float32), \
               tf.cast(progress_array[start_idx: start_idx + trajectory_steps], dtype=tf.float32)
        # tf.cast(final_idx, dtype=tf.int32),

    # option_no_order = tf.data.Options()
    # option_no_order.experimental_deterministic = False

    # filenames = tf.io.gfile.glob(path + "*.tfrec")
    # test_dataset = tf.data.TFRecordDataset(filenames)
    # count = 0
    # for item in test_dataset:
    #     foo = read_tfrecord(item)
    #     foo = merge_actions_rl(*foo)
    #     count += 1

    filenames = tf.io.gfile.glob(path + "*.tfrec")
    # filenames_ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    # filenames_ds = filenames_ds.shuffle(len(filenames), reshuffle_each_iteration=True)
    filenames_ds = tf.data.Dataset.list_files(filenames)
    # filenames_ds = filenames_ds.repeat(100)

    ds = filenames_ds.interleave(lambda x: tf.data.TFRecordDataset(x),
                                 cycle_length=5,
                                 num_parallel_calls=AUTO
                                 )
    ds = ds.map(read_tfrecord, num_parallel_calls=AUTO)
    if model_name == "actor_critic_residual_six_actions":
        ds = ds.map(merge_actions_rl, num_parallel_calls=AUTO)
    else:
        raise NotImplementedError
    return ds


def read_records_for_rl_pg(feature_maps_shape, actions_shape, model_name, path,
                           filenames=None, amplify_probs=False):
    # read from TFRecords. For optimal performance, read from multiple
    # TFRecord files at once and set the option experimental_deterministic = False
    # to allow order-altering optimizations.

    def read_tfrecord(example):
        features = {
            "action_numbers": tf.io.FixedLenFeature([], tf.string),
            "action_probs_1": tf.io.FixedLenFeature([], tf.string),
            "action_probs_2": tf.io.FixedLenFeature([], tf.string),
            "action_probs_3": tf.io.FixedLenFeature([], tf.string),
            "observation": tf.io.FixedLenFeature([], tf.string),
            "reward": tf.io.FixedLenFeature([], tf.float32),
            "progress_value": tf.io.FixedLenFeature([], tf.float32),
        }
        # decode the TFRecord
        example = tf.io.parse_single_example(example, features)

        action_numbers = tf.io.parse_tensor(example["action_numbers"], tf.int16)
        action_numbers = tf.cast(action_numbers, dtype=tf.int32)
        action_numbers.set_shape(len(actions_shape))

        action_probs_1 = tf.io.parse_tensor(example["action_probs_1"], tf.float16)
        action_probs_1 = tf.cast(action_probs_1, dtype=tf.float32)
        action_probs_1.set_shape(actions_shape[0][0])
        action_probs_2 = tf.io.parse_tensor(example["action_probs_2"], tf.float16)
        action_probs_2 = tf.cast(action_probs_2, dtype=tf.float32)
        action_probs_2.set_shape(actions_shape[1][0])
        action_probs_3 = tf.io.parse_tensor(example["action_probs_3"], tf.float16)
        action_probs_3 = tf.cast(action_probs_3, dtype=tf.float32)
        action_probs_3.set_shape(actions_shape[2][0])

        observation = tf.io.parse_tensor(example["observation"], tf.string)
        observation = tf.expand_dims(observation, axis=0)
        observation = tf.io.deserialize_many_sparse(observation, dtype=tf.float16)
        observation = tf.sparse.to_dense(observation)
        observation = tf.squeeze(observation)
        observation = tf.cast(observation, dtype=tf.float32)
        observation.set_shape(feature_maps_shape)

        reward = example["reward"]
        reward.set_shape(())
        reward = tf.cast(reward, dtype=tf.float32)

        progress_value = example["progress_value"]
        progress_value.set_shape(())
        progress_value = tf.cast(progress_value, dtype=tf.float32)

        return action_numbers, action_probs_1, action_probs_2, action_probs_3, observation, reward, progress_value

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    if filenames is None:
        filenames = tf.io.gfile.glob(path + "*.tfrec")

    # test_dataset = tf.data.Dataset.list_files(filenames)
    # test_dataset = test_dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
    #                                        cycle_length=5,
    #                                        num_parallel_calls=AUTO,
    #                                        )
    # count = 0
    # for item in test_dataset:
    #     foo = read_tfrecord(item)
    #     foo = random_reverse_pg(*foo)
    #     foo = merge_actions_pg(*foo)
    #     count += 1

    # filenames_ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    filenames_ds = tf.data.Dataset.list_files(filenames)
    # filenames_ds = filenames_ds.with_options(option_no_order)
    ds = filenames_ds.interleave(lambda x: tf.data.TFRecordDataset(x),
                                 cycle_length=5,
                                 num_parallel_calls=AUTO
                                 )
    ds = ds.map(read_tfrecord, num_parallel_calls=AUTO)
    ds = ds.map(random_reverse_pg, num_parallel_calls=AUTO)
    if model_name == "actor_critic_residual_shrub":
        raise NotImplementedError
    elif model_name == "actor_critic_residual_six_actions":
        ds = ds.map(merge_actions_pg, num_parallel_calls=AUTO)
    else:
        raise NotImplementedError
    ds = ds.shuffle(10000)
    return ds
