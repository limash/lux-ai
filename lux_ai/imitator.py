import abc
import time
import pickle

import ray
import numpy as np
import tensorflow as tf

from lux_ai import models, tools, tfrecords_storage
from lux_gym.envs.lux.action_vectors_new import empty_worker_action_vectors


class Agent(abc.ABC):
    def __init__(self, config, data, global_var_actor=None, filenames=None, current_cycle=None):

        self._feature_maps_shape = tools.get_feature_maps_shape(config["environment"])
        self._actions_shape = [item.shape for item in empty_worker_action_vectors]
        self._model_name = config["model_name"]
        self._batch_size = config["batch_size"]
        if self._model_name == "actor_critic_residual_shrub":
            self._model = models.actor_critic_residual_shrub(self._actions_shape)
            self._loss_function1 = tools.LossFunction1()
            self._loss_function2_0 = tools.LossFunction2(tf.constant([self._batch_size], dtype=tf.int64))
            self._loss_function2_1 = tools.LossFunction2(tf.constant([self._batch_size], dtype=tf.int64))
            self._loss_function3 = tools.LossFunction3(tf.constant([self._batch_size], dtype=tf.int64))
        elif self._model_name == "actor_critic_residual_with_transfer":
            self._model = models.actor_critic_residual_with_transfer()
            self._loss_function1 = tools.LossFunctionSevenActions()
            self._loss_function2 = tools.LossFunction2(tf.constant([self._batch_size], dtype=tf.int64))
            self._loss_function3 = tools.LossFunction3(tf.constant([self._batch_size], dtype=tf.int64))
        elif self._model_name == "actor_critic_residual_six_actions":
            self._model = models.actor_critic_residual_six_actions(6)
            self._loss_function = tools.LossFunctionSixActions()
        elif self._model_name == "actor_critic_efficient_six_actions":
            self._model = models.actor_critic_efficient_six_actions(6)
            self._loss_function = tools.LossFunctionSixActions()
        else:
            raise NotImplementedError

        # launch a model once to define structure
        dummy_feature_maps = np.zeros(self._feature_maps_shape, dtype=np.float32)
        dummy_feature_maps[6, 6, :1] = 1
        dummy_input = tf.convert_to_tensor(dummy_feature_maps, dtype=tf.float32)
        dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
        self._model(dummy_input)
        if data is not None:
            self._model.set_weights(data['weights'])
            print("Continue the model training.")

        self._global_var_actor = global_var_actor
        self._filenames = filenames
        self._current_cycle = current_cycle

        # self._dataset = reverb.TrajectoryDataset.from_table_signature(
        #     server_address=f'localhost:{buffer_server_port}',
        #     table=buffer_table_names[0],
        #     max_in_flight_samples_per_worker=2*config["batch_size"])
        # self._client = reverb.Client(f'localhost:{buffer_server_port}')

    def imitate(self):
        ds_train = tfrecords_storage.read_records_for_imitator(self._feature_maps_shape, self._actions_shape,
                                                               self._model_name,
                                                               "data/tfrecords/imitator/train/")
        ds_train = ds_train.batch(self._batch_size).prefetch(1)  # , drop_remainder=True)

        # for sample in ds_train:
        #     observations = sample[0].numpy()
        #     actions_probs0 = sample[1][0].numpy()
        #     actions_probs1 = sample[1][1].numpy()
        #     actions_probs2 = sample[1][2].numpy()
        #     print(np.sum(actions_probs0, axis=0))
        #     # actions_probs3 = sample[1][3].numpy()
        #     # total_rewards = sample[1][1].numpy()
        #     probs_output0, probs_output1, probs_output2, value_output = self._model(observations)
        #     # probs_output, value_output = self._model(observations)
        #     probs_output_v = probs_output0.numpy()
        #     real = np.argmax(actions_probs0, axis=1)
        #     preds = np.argmax(probs_output_v, axis=1)
        #     foo = (real == preds)
        #     print(np.sum(foo)/self._batch_size)
        #     # skewed_loss = self._loss_function(sample[1][0], probs_output)
        #     skewed_loss1 = self._loss_function1(sample[1][0], probs_output0)
        #     skewed_loss2 = self._loss_function2(sample[1][1], probs_output1)
        #     skewed_loss3 = self._loss_function3(sample[1][2], probs_output2)
        #     # skewed_loss4 = self._loss_function3(sample[1][3], probs_output3)

        if self._model_name == "actor_critic_residual_shrub":
            self._model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss={
                    "output_1": self._loss_function1,
                    "output_2": self._loss_function2_0,
                    "output_3": self._loss_function2_1,
                    "output_4": self._loss_function3,
                    "output_5": None  # tf.keras.losses.MeanSquaredError()
                },
                metrics={
                    "output_1": [tf.keras.metrics.CategoricalAccuracy()],
                },
            )
        elif self._model_name == "actor_critic_residual_with_transfer":
            self._model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss={
                    "output_1": self._loss_function1,
                    "output_2": self._loss_function2,
                    "output_3": self._loss_function3,
                    "output_4": None,
                },
                metrics={
                    "output_1": [tf.keras.metrics.CategoricalAccuracy()],
                },
            )
        elif self._model_name == "actor_critic_residual_six_actions":
            self._model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                loss={
                    "output_1": self._loss_function,
                    "output_2": None,
                },
                metrics={
                    "output_1": [tf.keras.metrics.CategoricalAccuracy()],
                },
            )
        elif self._model_name == "actor_critic_efficient_six_actions":
            self._model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                loss={
                    "output_1": self._loss_function,
                    "output_2": None,
                },
                metrics={
                    "output_1": [tf.keras.metrics.CategoricalAccuracy()],
                },
            )
        else:
            raise NotImplementedError

        save_weights_callback = tools.SaveWeightsCallback()
        # lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2, verbose=1)
        # early_stop_callback = tf.keras.callbacks.EarlyStopping(
        #     monitor='val_output_1_loss',
        #     patience=10,
        #     restore_best_weights=False,
        # )
        self._model.fit(ds_train, epochs=10,  # validation_data=ds_valid,
                        verbose=2,
                        callbacks=[save_weights_callback, ]
                        )
        weights = self._model.get_weights()
        data = {
            'weights': weights,
        }
        with open(f'data/data.pickle', 'wb') as f:
            pickle.dump(data, f, protocol=4)

        if self._global_var_actor is not None:
            ray.get(self._global_var_actor.set_done.remote(True))
        print("Imitation is done.")
        time.sleep(1)

    def self_imitate(self):
        ds_train = tfrecords_storage.read_records_for_imitator(self._feature_maps_shape, self._actions_shape,
                                                               self._model_name,
                                                               "_0^0_",
                                                               filenames=self._filenames,
                                                               amplify_probs=True)
        ds_train = ds_train.batch(self._batch_size).prefetch(1)  # , drop_remainder=True)

        # for sample in ds_train.take(10):
        #     observations = sample[0].numpy()
        #     actions_probs0 = sample[1][0].numpy()
        #     total_rewards = sample[1][1].numpy()
        #     probs_output, value_output = self._model(observations)
        #     probs_output_v = probs_output.numpy()

        if self._model_name == "actor_critic_residual_shrub":
            self._model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss={
                    "output_1": self._loss_function1,
                    "output_2": self._loss_function2_0,
                    "output_3": self._loss_function2_1,
                    "output_4": self._loss_function3,
                    "output_5": None  # tf.keras.losses.MeanSquaredError()
                },
                metrics={
                    "output_1": [tf.keras.metrics.CategoricalAccuracy()],
                },
            )
        elif self._model_name == "actor_critic_residual_six_actions":
            self._model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                loss={
                    "output_1": self._loss_function,
                    "output_2": None,
                },
                metrics={
                    "output_1": [tf.keras.metrics.CategoricalAccuracy()],
                },
            )
        else:
            raise NotImplementedError

        stop_training = tools.StopOnDemand(self._global_var_actor)
        self._model.fit(ds_train, epochs=10,  # validation_data=ds_valid,
                        verbose=2,
                        callbacks=[stop_training, ]
                        )
        weights = self._model.get_weights()
        data = {
            'weights': weights,
        }
        with open(f'data/weights/{self._current_cycle}.pickle', 'wb') as f:
            pickle.dump(data, f, protocol=4)
        print(f"{self._current_cycle}.pickle is saved.")
        print("Imitation is done.")
        time.sleep(1)
