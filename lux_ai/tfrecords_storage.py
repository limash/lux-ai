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


def to_tfrecord(reward, action_probs, action_mask, observation):
    feature = {
        "observation": _bytestring_feature([observation]),
        "action_mask": _bytestring_feature([action_mask]),
        "action_probs": _bytestring_feature([action_probs]),
        "reward": _float_feature([reward]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def write_tfrecord(ds, record_number, record_name):
    filename = f"data/tfrecords/imitator/train/{record_name}.tfrec"

    with tf.io.TFRecordWriter(filename) as out_file:
        for n, (observation, action_mask, action_probs, reward) in enumerate(ds):
            serial_action_probs = tf.io.serialize_tensor(action_probs)
            serial_action_mask = tf.io.serialize_tensor(action_mask)
            serial_observation = tf.io.serialize_tensor(observation)
            example = to_tfrecord(reward.numpy().astype(np.float32),
                                  serial_action_probs.numpy(),
                                  serial_action_mask.numpy(),
                                  serial_observation.numpy())
            out_file.write(example.SerializeToString())
        print(f"Wrote file #{record_number} {record_name}.tfrec containing {n} records")


def record_for_imitator(player1_data, player2_data, final_reward_1, final_reward_2,
                        feature_maps_shape, actions_number, record_number, record_name):

    def data_gen_all():
        for j, player_data in enumerate((player1_data, player2_data)):
            if player_data is None:
                continue
            final_reward = final_reward_1 if j == 0 else final_reward_2
            for key, unit in player_data.items():
                # unit_type = key.split("_")[0]
                # if unit_type != "u":
                #     continue
                actions = unit.actions
                counters = np.zeros_like(actions)  # for debug, shows points with actions yielded
                for point in unit.data:
                    # point [action, action_probs, actions_mask, observation]
                    action = point[0]
                    # store observation, action_mask, action_probs, reward
                    yield point[3], point[2], point[1], final_reward
                    counters += action

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
                median = np.median(unit.actions[np.nonzero(unit.actions)])
                actions = unit.actions
                multipliers = np.divide(median, actions, out=np.zeros_like(actions), where=actions != 0)
                final_idx = len(unit.data)
                counters = np.zeros_like(actions)  # for debug, shows points with actions yielded
                if np.nonzero(unit.actions)[0].shape[0] == 1:
                    # if only one possible action in unit episode trajectory, add only the last one (issue?)
                    idx = final_idx - 1
                else:
                    idx = 0
                while True:
                    # point [action, action_probs, actions_mask, observation]
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

                    # store observation, action_mask, action_probs, reward
                    yield point[3], point[2], point[1], final_reward
                    counters += action

                    idx += 1
                    if idx == final_idx:
                        break

    # result = []
    # generator = data_gen()
    # while len(result) < 10:
    #     x = next(generator)
    #     result.append(x)

    dataset = tf.data.Dataset.from_generator(
        data_gen_soft,
        output_signature=(
            tf.TensorSpec(shape=feature_maps_shape, dtype=tf.float16),
            tf.TensorSpec(shape=actions_number, dtype=tf.float16),
            tf.TensorSpec(shape=actions_number, dtype=tf.float16),
            tf.TensorSpec(shape=(), dtype=tf.float16),
        ))

    # foo = list(dataset.take(1))
    write_tfrecord(dataset, record_number, record_name)


def read_records_for_imitator(feature_maps_shape, actions_shape, path):
    # read from TFRecords. For optimal performance, read from multiple
    # TFRecord files at once and set the option experimental_deterministic = False
    # to allow order-altering optimizations.

    def read_tfrecord(example):
        features = {
            "observation": tf.io.FixedLenFeature([], tf.string),
            "action_mask": tf.io.FixedLenFeature([], tf.string),
            "action_probs": tf.io.FixedLenFeature([], tf.string),
            "reward": tf.io.FixedLenFeature([], tf.float32),
        }
        # decode the TFRecord
        example = tf.io.parse_single_example(example, features)

        observation = tf.io.parse_tensor(example["observation"], tf.float16)
        observation.set_shape(feature_maps_shape)
        action_mask = tf.io.parse_tensor(example["action_mask"], tf.float16)
        action_mask.set_shape(actions_shape)
        action_probs = tf.io.parse_tensor(example["action_probs"], tf.float16)
        action_probs.set_shape(actions_shape)
        reward = example["reward"]
        reward.set_shape(())

        return observation, action_mask, action_probs, reward

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    filenames = tf.io.gfile.glob(path + "*.tfrec")
    filenames_ds = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    filenames_ds = filenames_ds.with_options(option_no_order)
    ds = filenames_ds.map(read_tfrecord, num_parallel_calls=AUTO)
    ds = ds.shuffle(1000)
    return ds
