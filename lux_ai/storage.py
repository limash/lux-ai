from typing import List

import reverb
import tensorflow as tf

from lux_gym.envs.lux.action_vectors import action_vector


physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def send_data(player_data, total_reward, progress,
              feature_maps_shape, actions_number, n_points,
              client, table_names):
    obs_zeros = tf.zeros(feature_maps_shape, dtype=tf.float16)
    act_zeros = tf.zeros(actions_number, dtype=tf.float16)
    act_ones = tf.ones(actions_number, dtype=tf.float16)
    act_probs_uni = tf.ones(actions_number, dtype=tf.float16) * 1 / actions_number

    for data_object in player_data.values():
        entity_temporal_data_list, current_step = data_object.data, data_object.step
        entity_temporal_data_list = tf.nest.map_structure(lambda x: tf.convert_to_tensor(x, dtype=tf.float16),
                                                          entity_temporal_data_list)
        with client.trajectory_writer(num_keep_alive_refs=n_points) as writer:
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
                if i >= n_points - 1:
                    writer.create_item(
                        table=table_names[0],
                        priority=1.5,
                        trajectory={
                            'actions': writer.history['action'][-n_points:],
                            'actions_probs': writer.history['action_probs'][-n_points:],
                            'actions_masks': writer.history['action_mask'][-n_points:],
                            'observations': writer.history['observation'][-n_points:],
                            'total_rewards': writer.history['total_reward'][-n_points:],
                            'temporal_masks': writer.history['temporal_mask'][-n_points:],
                            'progresses': writer.history['progress'][-n_points:],
                        }
                    )
            i += 1
            for j in range(i, i + n_points - 1):
                writer.append({'action': act_zeros,
                               'action_probs': act_probs_uni,
                               'action_mask': act_ones,
                               'observation': obs_zeros,
                               'total_reward': tf.constant(0, dtype=tf.float16),
                               'temporal_mask': tf.constant(0, dtype=tf.float16),
                               'progress': tf.constant(1, dtype=tf.float16)
                               })
                if j >= n_points - 1:
                    writer.create_item(
                        table=table_names[0],
                        priority=1.5,
                        trajectory={
                            'actions': writer.history['action'][-n_points:],
                            'actions_probs': writer.history['action_probs'][-n_points:],
                            'actions_masks': writer.history['action_mask'][-n_points:],
                            'observations': writer.history['observation'][-n_points:],
                            'total_rewards': writer.history['total_reward'][-n_points:],
                            'temporal_masks': writer.history['temporal_mask'][-n_points:],
                            'progresses': writer.history['progress'][-n_points:],
                        }
                    )
            writer.end_episode()


class UniformBuffer:
    def __init__(self,
                 observations_shape,
                 num_tables: int = 1,
                 min_size: int = 64,
                 max_size: int = 100000,
                 n_points: int = 2,
                 checkpointer=None):

        OBSERVATION_SPEC = tf.TensorSpec(observations_shape, tf.float16)
        ACTION_SPEC = tf.TensorSpec([len(action_vector)], tf.float16)

        self._min_size = min_size
        self._table_names = [f"uniform_table_{i}" for i in range(num_tables)]
        self._server = reverb.Server(
            tables=[
                reverb.Table(
                    name=self._table_names[i],
                    sampler=reverb.selectors.Uniform(),
                    remover=reverb.selectors.Fifo(),
                    max_size=int(max_size),
                    rate_limiter=reverb.rate_limiters.MinSize(min_size),
                    signature={
                        'actions': tf.TensorSpec([n_points, *ACTION_SPEC.shape], ACTION_SPEC.dtype),
                        'actions_probs': tf.TensorSpec([n_points, *ACTION_SPEC.shape], ACTION_SPEC.dtype),
                        'actions_masks': tf.TensorSpec([n_points, *ACTION_SPEC.shape], ACTION_SPEC.dtype),
                        'observations': tf.TensorSpec([n_points, *OBSERVATION_SPEC.shape], OBSERVATION_SPEC.dtype),
                        'total_rewards': tf.TensorSpec([n_points], tf.float16),
                        'temporal_masks': tf.TensorSpec([n_points], tf.float16),
                        'progresses': tf.TensorSpec([n_points], tf.float16),
                    }
                ) for i in range(num_tables)
            ],
            # Sets the port to None to make the server pick one automatically.
            port=None,
            checkpointer=checkpointer
        )

    @property
    def table_names(self) -> List[str]:
        return self._table_names

    @property
    def min_size(self) -> int:
        return self._min_size

    @property
    def server_port(self) -> int:
        return self._server.port
