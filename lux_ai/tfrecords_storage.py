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


def write_tfrecord(ds, record_number, record_name, is_for_rl, save_path=None):
    if is_for_rl:
        if save_path is None:
            save_path = "data/tfrecords/rl/storage/"
        # with tf.io.TFRecordWriter(filename) as out_file:
        out_file = None
        for n, (actions_numbers, actions_probs, observations, rewards,
                masks, progress_array, final_idx) in enumerate(ds):
            if out_file is None or n % 10 == 0:
                if out_file is not None:
                    out_file.close()
                filename = f"{save_path}{record_name}_{n}.tfrec"
                out_file = tf.io.TFRecordWriter(filename)
            s_actions_numbers = tf.io.serialize_tensor(actions_numbers)
            # s_actions_probs = tf.io.serialize_tensor(actions_probs)
            serial_action_probs = [tf.io.serialize_tensor(item).numpy() for item in actions_probs]
            s_observations = tf.io.serialize_tensor(tf.io.serialize_sparse(observations))
            s_rewards = tf.io.serialize_tensor(rewards)
            s_masks = tf.io.serialize_tensor(masks)
            s_progress_array = tf.io.serialize_tensor(progress_array)
            # s_final_idx = tf.io.serialize_tensor(final_idx)
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
    else:
        if save_path is None:
            save_path = "data/tfrecords/imitator/train/"
        filename = f"{save_path}{record_name}.tfrec"
        with tf.io.TFRecordWriter(filename) as out_file:
            for n, (observation, action_probs, reward) in enumerate(ds):
                serial_action_probs = [tf.io.serialize_tensor(item).numpy() for item in action_probs]
                serial_observation = tf.io.serialize_tensor(tf.io.serialize_sparse(observation))
                example = to_tfrecord(reward.numpy().astype(np.float32),
                                      serial_action_probs,
                                      serial_observation.numpy())
                out_file.write(example.SerializeToString())
        print(f"Wrote file #{record_number} {record_name}.tfrec containing {n} records")


def record(player1_data, player2_data, final_reward_1, final_reward_2,
           feature_maps_shape, actions_shape, record_number, record_name,
           progress=None, is_for_rl=False, save_path=None):
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
                # actions = unit.actions
                # counters = np.zeros_like(actions)  # for debug, shows points with actions yielded
                # create np arrays to store data
                # actions_numbers = -np.ones([total_len])
                # actions_probs = np.zeros([total_len] + list(actions.shape))
                actions_numbers = -np.ones([total_len] + [len(unit.data[0][1])])
                actions_probs = [np.zeros([total_len] + list(item.shape)) for item in unit.data[0][1]]
                observations = np.zeros([total_len] + list(unit.data[0][2].shape))
                rewards = np.zeros([total_len])
                masks = np.zeros([total_len])
                progress_array = np.zeros([total_len])
                i = 0
                for i, point in enumerate(unit.data):
                    # point [action, action_probs, observation]
                    # actions_numbers[i] = np.argmax(point[0])
                    # actions_probs[i] = point[1]
                    # actions_numbers[i] = np.array([np.argmax(item) for item in point[0]])
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
                    # counters += point[0]
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
        """Generator, which softens very skewed distribution of actions
        repeating rare and skipping very often actions"""
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

    # result = []
    # generator = data_gen_all_for_rl()
    # while len(result) < 1000:
    #     x = next(generator)
    #     result.append(x)

    if is_for_rl:
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
    write_tfrecord(dataset, record_number, record_name, is_for_rl, save_path)


def up_down(obs, probs):
    obs = tf.reverse(obs, [0])
    if probs[0][0] == 1 or probs[0][1] == 1:
        if probs[1][0] == 1:
            probs[1] = tf.constant([0, 0, 1, 0], dtype=tf.float32)
        elif probs[1][2] == 1:
            probs[1] = tf.constant([1, 0, 0, 0], dtype=tf.float32)
    return obs, probs


def left_right(obs, probs):
    obs = tf.reverse(obs, [1])
    if probs[0][0] == 1 or probs[0][1] == 1:
        if probs[1][1] == 1:
            probs[1] = tf.constant([0, 0, 0, 1], dtype=tf.float32)
        elif probs[1][3] == 1:
            probs[1] = tf.constant([0, 1, 0, 0], dtype=tf.float32)
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
    if act_probs[0] == 1:  # movement action
        movements = dir_probs
    else:
        movements = tf.constant([0, 0, 0, 0], dtype=tf.float32)
    if act_probs[1] == 1 or act_probs[2] == 1:  # transfer of idle
        idle = tf.constant([1, ], dtype=tf.float32)
    else:
        idle = tf.constant([0, ], dtype=tf.float32)
    if act_probs[3] == 1:
        bcity = tf.constant([1, ], dtype=tf.float32)
    else:
        bcity = tf.constant([0, ], dtype=tf.float32)
    new_actions = tf.concat([movements, idle, bcity], axis=0)
    return observation, (new_actions, reward)


def merge_actions_rl(act_numbers, act_probs, dir_probs, res_probs, observation, reward, mask, progress):
    # steps = act_probs.shape[0]
    steps = 40
    ta = tf.TensorArray(dtype=tf.float32, size=steps, dynamic_size=False)
    for i in tf.range(steps):
        if act_probs[i][0] == 1:  # movement action
            movements = dir_probs[i]
        else:
            movements = tf.constant([0, 0, 0, 0], dtype=tf.float32)
        if act_probs[i][1] == 1 or act_probs[i][2] == 1:  # transfer of idle
            idle = tf.constant([1, ], dtype=tf.float32)
        else:
            idle = tf.constant([0, ], dtype=tf.float32)
        if act_probs[i][3] == 1:
            bcity = tf.constant([1, ], dtype=tf.float32)
        else:
            bcity = tf.constant([0, ], dtype=tf.float32)
        row = tf.concat([movements, idle, bcity], axis=0)
        ta = ta.write(i, row)
    new_probs = ta.stack()
    act_numbers = tf.argmax(new_probs, axis=1)
    return act_numbers, new_probs, observation, reward, mask, progress


def read_records_for_imitator(feature_maps_shape, actions_shape, model_name, path):
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

    filenames = tf.io.gfile.glob(path + "*.tfrec")

    # test_dataset = tf.data.Dataset.from_tensor_slices(filenames[:12])
    # test_dataset = tf.data.Dataset.list_files(filenames[:12])
    # test_dataset = test_dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
    #                                        cycle_length=5,
    #                                        num_parallel_calls=AUTO,
    #                                        )
    # test_dataset = tf.data.TFRecordDataset(filenames[:12])
    # count = 0
    # for item in test_dataset:
    #     foo = read_tfrecord(item)
    #     # foo = random_reverse(*foo)
    #     # foo = split_movement_actions(*foo)
    #     # foo = merge_actions(*foo)
    #     count += 1

    filenames_ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    # filenames_ds = tf.data.Dataset.list_files(filenames)
    filenames_ds = filenames_ds.with_options(option_no_order)
    # ds = filenames_ds.interleave(lambda x: tf.data.TFRecordDataset(x),
    #                              cycle_length=5,
    #                              num_parallel_calls=AUTO
    #                              )
    ds = filenames_ds.map(read_tfrecord, num_parallel_calls=AUTO)
    ds = ds.map(random_reverse, num_parallel_calls=AUTO)
    if model_name == "actor_critic_residual_shrub":
        ds = ds.map(split_movement_actions, num_parallel_calls=AUTO)
    elif model_name == "actor_critic_residual_six_actions":
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
        start_idx = tf.random.uniform(shape=(), minval=0, maxval=final_idx, dtype=tf.int64)

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

    # test_dataset = tf.data.TFRecordDataset(path + '29588162_Toad Brigade.tfrec')
    # count = 0
    # for item in test_dataset:
    #     foo = read_tfrecord(item)
    #     foo = merge_actions_rl(*foo)
    #     count += 1

    filenames = tf.io.gfile.glob(path + "*.tfrec")
    filenames_ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    filenames_ds = filenames_ds.shuffle(len(filenames), reshuffle_each_iteration=True)
    filenames_ds = filenames_ds.repeat(episode_length)

    # filenames_ds = filenames_ds.with_options(option_no_order)
    ds = filenames_ds.map(read_tfrecord, num_parallel_calls=AUTO)
    if model_name == "actor_critic_residual_six_actions":
        ds = ds.map(merge_actions_rl, num_parallel_calls=AUTO)
    else:
        raise NotImplementedError
    return ds
