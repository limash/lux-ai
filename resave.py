import numpy as np
import tensorflow as tf

load_path = "data/tfrecords/imitator/train/"
features = {
    "observation": tf.io.FixedLenFeature([], tf.string),
    "action_probs_1": tf.io.FixedLenFeature([], tf.string),
    "action_probs_2": tf.io.FixedLenFeature([], tf.string),
    "action_probs_3": tf.io.FixedLenFeature([], tf.string),
    "reward": tf.io.FixedLenFeature([], tf.float32),
}


def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def _float_feature(list_of_floats):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def to_tfrecord(reward, action_probs, observation):
    feature = {
        "observation": _bytestring_feature([observation]),
        "action_probs_1": _bytestring_feature([action_probs[0]]),
        "action_probs_2": _bytestring_feature([action_probs[1]]),
        "action_probs_3": _bytestring_feature([action_probs[2]]),
        "reward": _float_feature([reward]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def resave():
    filenames = tf.io.gfile.glob(load_path + "*.tfrec")
    filenames_ds = tf.data.TFRecordDataset(filenames)
    iterator = iter(filenames_ds)

    j = 0
    while True:
        filename = f"data/tfrecords/imitator/validation/{j}.tfrec"
        with tf.io.TFRecordWriter(filename) as out_file:
            i = 0
            while True:
                # reading
                try:
                    example = next(iterator)
                    example = tf.io.parse_single_example(example, features)
                    observation = tf.io.parse_tensor(example["observation"], tf.string)
                    observation = tf.expand_dims(observation, axis=0)
                    observation = tf.io.deserialize_many_sparse(observation, dtype=tf.float16)
                    action_probs_1 = tf.io.parse_tensor(example["action_probs_1"], tf.float16)
                    action_probs_2 = tf.io.parse_tensor(example["action_probs_2"], tf.float16)
                    action_probs_3 = tf.io.parse_tensor(example["action_probs_3"], tf.float16)
                    reward = example["reward"]
                    action_probs = action_probs_1, action_probs_2, action_probs_3
                    serial_action_probs = [tf.io.serialize_tensor(item).numpy() for item in action_probs]
                    serial_observation = tf.io.serialize_tensor(tf.io.serialize_sparse(observation))
                    example = to_tfrecord(reward.numpy().astype(np.float32),
                                          serial_action_probs,
                                          serial_observation.numpy())
                    out_file.write(example.SerializeToString())
                    if i > 5000:
                        break
                    i += 1
                except StopIteration:
                    return
        j += 1


if __name__ == '__main__':
    resave()
