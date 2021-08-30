import tensorflow as tf
import ray
import gym


def get_feature_maps_shape(env_name):
    environment = gym.make(env_name)
    _, observations = environment.reset_process()
    feature_maps_shape = observations[0]["workers"]["u_1"].shape
    return feature_maps_shape


def get_prob_logs_from_logits(logits, actions, n_outputs):
    probs = tf.nn.softmax(logits)
    mask = tf.one_hot(actions, n_outputs, dtype=tf.float32)
    masked_probs = tf.reduce_sum(probs * mask, axis=-1)
    logs = tf.math.log(tf.clip_by_value(masked_probs, 1.e-32, 1.))  # inappropriate values will be masked
    return logs


def prepare_td_lambda(values, returns, rewards, lmb, gamma):
    from collections import deque

    target_values = deque([returns])
    for i in range(values.shape[0] - 1, 0, -1):
        reward = rewards[i, :] if rewards is not None else 0
        target_values.appendleft(reward + gamma * ((1 - lmb) * values[i, :] + lmb * target_values[0]))

    target_values = tf.stack(tuple(target_values))
    return target_values


def tf_prepare_td_lambda_no_rewards(values, returns, lmb, gamma):
    reward = 0
    ta = tf.TensorArray(dtype=tf.float32, size=values.shape[1], dynamic_size=False)
    row = returns
    ta = ta.write(values.shape[1] - 1, row)
    for i in tf.range(values.shape[1] - 1, 0, -1):
        # prev = ta.read(i)  # read does not work properly
        # row = reward + gamma * ((1 - lmb) * values[i, :] + lmb * prev)
        row = reward + gamma * ((1 - lmb) * values[:, i] + lmb * row)
        ta = ta.write(i - 1, row)

    target_values = ta.stack()
    return tf.transpose(target_values)


def get_entropy(logits, mask=None):
    # another way to calculate entropy:
    # probs = tf.nn.softmax(logits)
    # entropy = tf.keras.losses.categorical_crossentropy(probs, probs)
    probs = tf.nn.softmax(logits)
    log_probs = tf.nn.log_softmax(logits)
    # log_probs = tf.math.log(probs)
    if mask is not None:
        probs = probs * mask
        log_probs = log_probs * mask
    entropy = tf.reduce_sum(-probs * log_probs, axis=-1)
    return entropy


@ray.remote
class GlobalVarActor:
    def __init__(self):
        self.global_v = 1
        self.current_weights = None, None
        self.done = False

    def set_global_v(self, v):
        self.global_v = v

    def get_global_v(self):
        return self.global_v

    def set_current_weights(self, w):
        self.current_weights = w

    def get_current_weights(self):
        return self.current_weights

    def set_done(self, done):
        self.done = done

    def get_done(self):
        return self.done


class DataValue:
    def __init__(self):
        self.data = []
        self.step = []

    def append(self, data, step):
        self.data.append(data)
        self.step.append(step)
