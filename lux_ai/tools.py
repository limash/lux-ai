import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
import ray
import gym

from lux_gym.envs import tools


# from lux_gym.envs.lux.action_vectors import worker_action_mask, cart_action_mask, citytile_action_mask

# actions_masks = (worker_action_mask, cart_action_mask, citytile_action_mask)


def squeeze_transform(obs_base, acts_rews):
    actions_probs, total_rewards = acts_rews
    observations = tools.squeeze(obs_base)
    return observations, (actions_probs, total_rewards)


def skewed_kldivergence_loss():
    def loss_function(y_true, y_pred):
        one = tf.constant(1.)
        # class_sums = tf.reduce_sum(y_true, axis=0)
        # movements_mean = tf.reduce_mean(class_sums[:4])
        # center_multiplier = movements_mean / class_sums[4]
        # if not tf.math.is_finite(center_multiplier):
        #     center_multiplier = one
        # build_multiplier = movements_mean / class_sums[5]
        # if not tf.math.is_finite(build_multiplier):
        #     build_multiplier = one
        center_multiplier = tf.constant(0.1)
        build_multiplier = tf.constant(2.)

        class_weights = tf.stack([one, one, one, one, center_multiplier, build_multiplier], axis=0)
        class_weights = tf.expand_dims(class_weights, axis=0)

        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true = backend.clip(y_true, backend.epsilon(), 1)
        y_pred = backend.clip(y_pred, backend.epsilon(), 1)
        # temp = y_true * tf.math.log(y_true / y_pred)
        # temp_v = temp.numpy()
        # temp_sum = tf.reduce_sum(temp, axis=-1)
        # result = temp * class_weight
        # result_v = result.numpy()
        # result_sum = tf.reduce_sum(result, axis=-1)
        return tf.reduce_sum(class_weights * y_true * tf.math.log(y_true / y_pred), axis=-1)

    return loss_function


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
            # point_value = [action, action_probs, actions_masks[i], observation]
            point_value = [action, action_probs, observation]
            if k1 in player_data.keys():
                player_data[k1].append(point_value, current_step, action[0])
            else:
                player_data[k1] = DataValue(action[0].shape[0])
                player_data[k1].append(point_value, current_step, action[0])
    return player_data


def merge_first_two_dimensions(input1, input2):
    (tensor1, tensor2), (tensor3, tensor4) = input1, input2
    tensors = tensor1, tensor2, tensor3, tensor4
    b = [tf.reshape(a, tf.concat([[tf.shape(a)[0] * tf.shape(a)[1]], tf.shape(a)[2:]], axis=0)) for a in tensors]
    tensor1, tensor2, tensor3, tensor4 = b
    outputs = (tensor1, tensor2), (tensor3, tensor4)
    return outputs


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


def get_prob_logs_from_probs(probs, actions, n_outputs):
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


def get_entropy_from_probs(probs, mask=None):
    log_probs = tf.math.log(tf.clip_by_value(probs, 1.e-32, 1.))
    if mask is not None:
        probs = probs * mask
        log_probs = log_probs * mask
    entropy = tf.reduce_sum(-probs * log_probs, axis=-1)
    return entropy


def norm_probs(probs_unnorm):
    # spec_probs_unnorm = layers.Multiply()([all_probs, input_B])
    # probs = layers.Lambda(norm_probs, name="probs_output")(spec_probs_unnorm)

    logits = tf.math.log(probs_unnorm)
    probs = tf.nn.softmax(logits)
    return probs


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
    def __init__(self, length):
        self.data = []
        self.step = []
        self.actions = np.zeros(length)

    def append(self, data, step, action):
        self.data.append(data)
        self.step.append(step)
        self.actions += action
