import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
import ray
import gym


# from lux_gym.envs.lux.action_vectors import worker_action_mask, cart_action_mask, citytile_action_mask

# actions_masks = (worker_action_mask, cart_action_mask, citytile_action_mask)


def squeeze_transform(obs_base, acts_probs):
    actions_probs, total_rewards = acts_probs
    features = obs_base
    # features_v = features.numpy()

    features_padded = tf.pad(features, tf.constant([[6, 6], [6, 6], [0, 0]]), mode="CONSTANT")
    # features_padded_v = features_padded.numpy()
    units_layers = features_padded[:, :, :1]
    units_coords = tf.cast(tf.where(units_layers), dtype=tf.int32)
    min_x = units_coords[:, 0] - 6
    max_x = units_coords[:, 0] + 6
    min_y = units_coords[:, 1] - 6
    max_y = units_coords[:, 1] + 6
    piece = features_padded[min_x[0]: max_x[0] + 1, min_y[0]: max_y[0] + 1, :]
    # piece_v = piece.numpy()

    features_padded_glob = tf.pad(features,
                                  tf.constant([[32, 32], [32, 32], [0, 0]]),
                                  mode="CONSTANT")
    # features_padded_glob_v = features_padded_glob.numpy()
    units_layers_glob = features_padded_glob[:, :, :1]
    units_coords_glob = tf.cast(tf.where(units_layers_glob), dtype=tf.int32)
    min_x_glob = units_coords_glob[:, 0] - 32
    max_x_glob = units_coords_glob[:, 0] + 32
    min_y_glob = units_coords_glob[:, 1] - 32
    max_y_glob = units_coords_glob[:, 1] + 32

    piece_glob1 = features_padded_glob[min_x_glob[0]: max_x_glob[0] + 1, min_y_glob[0]: max_y_glob[0] + 1, 45:49]
    piece_glob2 = features_padded_glob[min_x_glob[0]: max_x_glob[0] + 1, min_y_glob[0]: max_y_glob[0] + 1, 60:]
    piece_glob = tf.concat([piece_glob1, piece_glob2], axis=-1)
    # 17, 4; 5, 5
    pooled_piece_glob = tf.squeeze(tf.nn.avg_pool(tf.expand_dims(piece_glob, axis=0), 5, 5, padding="VALID"))
    # piece_glob_v = piece_glob.numpy()
    # pooled_piece_glob_v = pooled_piece_glob.numpy()

    piece_filtered = tf.concat([piece[:, :, :1],
                                piece[:, :, 4:10],
                                piece[:, :, 15:],
                                ], axis=-1)
    observations = tf.concat([piece_filtered, pooled_piece_glob], axis=-1)
    return observations, (actions_probs, total_rewards)


def skewed_kldivergence_loss(class_weight):
    def loss_function(y_true, y_pred):
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
        return tf.reduce_sum(class_weight * y_true * tf.math.log(y_true / y_pred), axis=-1)

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
                player_data[k1].append(point_value, current_step, action)
            else:
                player_data[k1] = DataValue(action.shape[0])
                player_data[k1].append(point_value, current_step, action)
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
