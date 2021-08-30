import tensorflow as tf

import reverb
from typing import List


class UniformBuffer:
    def __init__(self,
                 observations_shape,
                 num_tables: int = 1,
                 min_size: int = 64,
                 max_size: int = 100000,
                 n_points: int = 2,
                 checkpointer=None):

        OBSERVATION_SPEC = tf.TensorSpec(observations_shape, tf.float16)
        ACTION_SPEC = tf.TensorSpec([39], tf.float16)

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
