import abc
import time
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
# import reverb

from lux_ai import models, tools, tfrecords_storage
from lux_gym.envs.lux.action_vectors_new import empty_worker_action_vectors

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Agent(abc.ABC):
    def __init__(self, config, data):

        self._feature_maps_shape = tools.get_feature_maps_shape(config["environment"])
        self._actions_shape = [item.shape for item in empty_worker_action_vectors]
        if config["model_name"] == "actor_critic_residual":
            self._model = models.actor_critic_residual(self._actions_shape)
        elif config["model_name"] == "actor_critic_efficient":
            self._model = models.actor_critic_efficient(self._actions_shape)
        else:
            raise ValueError

        # launch a model once to define structure
        dummy_feature_maps = np.zeros(self._feature_maps_shape, dtype=np.float32)
        # dummy_feature_maps[16, 16, :1] = 1
        dummy_feature_maps[6, 6, :1] = 1
        dummy_input = tf.convert_to_tensor(dummy_feature_maps, dtype=tf.float32)
        dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
        self._model(dummy_input)

        self._batch_size = config["batch_size"]
        # class_weights = np.ones(self._actions_shape, dtype=np.single)
        # class_weights[4] = 0.1
        # class_weights[5] = 2.
        # class_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
        # self._class_weights = tf.expand_dims(class_weights, axis=0)
        # self._loss_function = tools.skewed_kldivergence_loss(self._class_weights)
        self._loss_function1 = tools.LossFunction1()
        self._loss_function2_0 = tools.LossFunction2(tf.constant([self._batch_size], dtype=tf.int64))
        self._loss_function2_1 = tools.LossFunction2(tf.constant([self._batch_size], dtype=tf.int64))
        self._loss_function3 = tools.LossFunction3(tf.constant([self._batch_size], dtype=tf.int64))

        if data is not None:
            self._model.set_weights(data['weights'])
            print("Continue the model training.")

        # self._dataset = reverb.TrajectoryDataset.from_table_signature(
        #     server_address=f'localhost:{buffer_server_port}',
        #     table=buffer_table_names[0],
        #     max_in_flight_samples_per_worker=2*config["batch_size"])
        # self._client = reverb.Client(f'localhost:{buffer_server_port}')

    def imitate(self):
        ds_train = tfrecords_storage.read_records_for_imitator(self._feature_maps_shape, self._actions_shape,
                                                               "data/tfrecords/imitator/train/")
        # ds_valid = tfrecords_storage.read_records_for_imitator(self._feature_maps_shape, self._actions_shape,
        #                                                        "data/tfrecords/imitator/validation/")
        ds_train = ds_train.batch(self._batch_size)  # , drop_remainder=True)
        # ds_valid = ds_valid.batch(self._batch_size)  # , drop_remainder=True)

        # for sample in ds_train.take(10):
        #     observations = sample[0].numpy()
        #     actions_probs0 = sample[1][0].numpy()
        #     actions_probs1 = sample[1][1].numpy()
        #     actions_probs2 = sample[1][2].numpy()
        #     actions_probs3 = sample[1][3].numpy()
        #     total_rewards = sample[1][4].numpy()
        #     probs_output0, probs_output1, probs_output2, probs_output3, value_output = self._model(observations)
        #     probs_output_v = probs_output0.numpy()
        #     value_output_v = value_output.numpy()
        #     skewed_loss1 = self._loss_function1(sample[1][0], probs_output0)
        #     skewed_loss2 = self._loss_function2_0(sample[1][1], probs_output1)
        #     skewed_loss3 = self._loss_function2_1(sample[1][2], probs_output2)
        #     skewed_loss4 = self._loss_function3(sample[1][3], probs_output3)

        # lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=1, verbose=1)
        # early_stop_callback = tf.keras.callbacks.EarlyStopping(
        #     monitor='val_loss',
        #     patience=2,
        #     restore_best_weights=True,
        # )

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss={
                "output_1": self._loss_function1,
                "output_2": self._loss_function2_0,
                "output_3": self._loss_function2_1,
                "output_4": self._loss_function3,
                "output_5": None  # tf.keras.losses.MeanSquaredError()
            },
            # metrics={
            #     "output_1": [tf.keras.metrics.CategoricalAccuracy()],
            #     # "value_output": [tf.keras.metrics.MeanAbsolutePercentageError(),
            #     #                  tf.keras.metrics.MeanAbsoluteError()]
            # },
            # loss_weights={"output_1": 2.0}  # ,
            # "output_2": 0.1},
        )

        self._model.fit(ds_train, epochs=1)
        # , validation_data=ds_valid, callbacks=[early_stop_callback, lr_scheduler])
        weights = self._model.get_weights()
        data = {
            'weights': weights,
        }
        with open(f'data/data.pickle', 'wb') as f:
            pickle.dump(data, f, protocol=4)

        print("Imitation is done.")


class ACAgent(abc.ABC):
    def __init__(self, config, data):

        self._feature_maps_shape = tools.get_feature_maps_shape(config["environment"])
        self._actions_shape = actions_number
        if config["model_name"] == "actor_critic_residual":
            self._model = models.actor_critic_residual(self._actions_shape)
        elif config["model_name"] == "actor_critic_efficient":
            self._model = models.actor_critic_efficient(self._actions_shape)
        else:
            raise ValueError

        # launch a model once to define structure
        dummy_feature_maps = np.zeros(self._feature_maps_shape, dtype=np.float32)
        dummy_feature_maps[6, 6, :1] = 1
        dummy_input = tf.convert_to_tensor(dummy_feature_maps, dtype=tf.float32)
        dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
        self._model(dummy_input)

        if data is not None:
            self._model.set_weights(data['weights'])
            print("Continue the model training.")

        self._batch_size = config["batch_size"]
        self._n_points = config["n_points"]
        self._iterations_number = config["iterations_number"]
        self._save_interval = config["save_interval"]

        self._optimizer = tfa.optimizers.AdamW(weight_decay=1.e-5, learning_rate=config["default_lr"])
        self._entropy_c = config["entropy_c"]
        self._entropy_c_decay = config["entropy_c_decay"]
        self._lambda = config["lambda"]

        self._is_debug = config["debug"]
        if not config["debug"]:
            self._training_step = tf.function(self._training_step)

    def _training_step(self, actions, behaviour_policy_probs, observations, total_rewards, masks, progress):
        print("Tracing")

        if self._is_debug:
            actions_v = actions.numpy()
            behaviour_policy_probs_v = behaviour_policy_probs.numpy()
            observations_v = observations.numpy()
            total_rewards_v = total_rewards.numpy()
            masks_v = masks.numpy()
            progress_v = progress.numpy()

        # actions = tf.transpose(actions)
        # behaviour_policy_logits = tf.transpose(behaviour_policy_logits, perm=[1, 0, 2])
        # maps = tf.transpose(observations[0], perm=[1, 0, 2, 3, 4])
        # scalars = tf.transpose(observations[1], perm=[1, 0, 2])
        # rewards = tf.transpose(rewards)
        # dones = tf.transpose(dones)

        # nsteps = tf.argmax(dones, axis=0, output_type=tf.int32)
        # ta = tf.TensorArray(dtype=tf.float32, size=self._sample_batch_size, dynamic_size=False)
        # for i in tf.range(self._sample_batch_size):
        #     row = tf.concat([tf.constant([0.]),
        #                      tf.linspace(0., 1., nsteps[i] + 1)[:-1],
        #                      tf.ones(steps - nsteps[i] - 1)], axis=0)
        #     ta = ta.write(i, row)
        # progress = ta.stack()
        # progress = tf.transpose(progress)

        # prepare a mask for the valid time steps
        # alive_positions = tf.where(actions != -1)
        # ones_array = tf.ones(alive_positions.shape[0])
        # mask = tf.scatter_nd(alive_positions, ones_array, actions.shape)
        # mask2d = tf.where(actions == -1, 0., 1.)
        # mask2d = tf.concat((tf.zeros([tf.shape(dones)[0], 1]), (tf.ones_like(dones) - dones)[:, :-1]), axis=1)
        # e_mask = tf.concat((tf.zeros([tf.shape(episode_dones)[0], 1]),
        #                     (tf.ones_like(episode_dones) - episode_dones)[:, :-1]), axis=1)
        # mask3d = tf.where(behaviour_policy_logits == 0., 0., 1.)
        mask2d = masks
        mask3d = tf.transpose(tf.ones([self._actions_shape, 1, 1], dtype=tf.float32) * mask2d, perm=[1, 2, 0])
        if self._is_debug:
            mask2d_v = mask2d.numpy()
            mask3d_v = mask3d.numpy()
        # e_mask_v = e_mask.numpy()

        # get final rewards, currently there is the only reward in the end of a game
        # returns = total_rewards[-1, :]

        # behaviour_action_log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=behaviour_policy_logits,
        #                                                                              labels=actions)
        # it is almost similar to above line, but above probably won't work on cpus (due to -1 actions)
        behaviour_action_log_probs = tools.get_prob_logs_from_probs(behaviour_policy_probs, actions,
                                                                    self._actions_shape)
        if self._is_debug:
            behaviour_action_log_probs_v = behaviour_action_log_probs.numpy()

        with tf.GradientTape() as tape:
            maps = observations
            if self._is_debug:
                maps_v = maps.numpy()
            # there are two ways to get outputs from the model
            # 1: using map_fn along the time dimension (or whatever), it is slow but consumes less memory
            # logits, values = tf.map_fn(self._model, (maps, scalars),
            #                            fn_output_signature=[tf.TensorSpec((self._sample_batch_size,
            #                                                                self._n_outputs), dtype=tf.float32),
            #                                                 tf.TensorSpec((self._sample_batch_size, 1),
            #                                                               dtype=tf.float32)])
            # -
            # 2: merging time and batch dimensions and applying the model at once, it is fast, but requires gpu memory
            maps_shape = tf.shape(maps)
            # scalars_shape = tf.shape(scalars)
            maps_merged = tf.reshape(maps, (-1, maps_shape[2], maps_shape[3], maps_shape[4]))
            # maps_merged = tf.reshape(maps, (-1, maps_shape[2], maps_shape[3]))
            # scalars_merged = tf.reshape(scalars, (-1, scalars_shape[2]))
            probs_merged, values_merged = self._model(maps_merged, training=True)
            probs = tf.reshape(probs_merged, (maps_shape[0], maps_shape[1], -1))
            values = tf.reshape(values_merged, (maps_shape[0], maps_shape[1], -1))
            # -

            # logits = tf.roll(logits, shift=1, axis=0)  # shift by 1 along time dimension, to match a pattern
            # values = tf.roll(values, shift=1, axis=0)  # where actions, logits, etc. led to the observation
            # probs = tf.roll(probs, shift=1, axis=1)  # shift by 1 along time dimension, to match a pattern
            # values = tf.roll(values, shift=1, axis=1)  # where actions, logits, etc. led to the observation
            target_action_log_probs = tools.get_prob_logs_from_probs(probs, actions, self._actions_shape)

            with tape.stop_recording():
                log_rhos = target_action_log_probs - behaviour_action_log_probs
                rhos = tf.exp(log_rhos)
                # rhos_masked = tf.where(actions == -1, 0., rhos)  # use where to remove nans, should be outside tape
                rhos_masked = rhos * mask2d
                clipped_rhos = tf.minimum(tf.constant(1.), rhos_masked)

            # add final rewards to 'empty' spots in values
            # values = tf.squeeze(values) * mask2d  # to ensure zeros in not valid spots
            # values = tf.where(e_mask == 0, total_rewards, values)  # to calculate targets
            values = tf.where(mask2d == 0, total_rewards, tf.squeeze(values))  # to calculate targets
            if self._is_debug:
                clipped_rhos_v = clipped_rhos.numpy()
                values_v = values.numpy()

            with tape.stop_recording():
                # calculate targets
                # targets = tools.prepare_td_lambda(tf.squeeze(values), returns, None, self._lambda, 1.)
                targets = tools.tf_prepare_td_lambda_no_rewards(values, total_rewards[:, 0], self._lambda, 1.)
                targets = targets * mask2d

            values = values * mask2d
            if self._is_debug:
                values_v = values.numpy()
                targets_v = targets.numpy()

            with tape.stop_recording():
                # td error with truncated IS weights (rhos), it is a constant:
                td_error = clipped_rhos * (targets - values)

            # critic loss
            # critic_loss = self._loss_fn(targets, values)
            critic_loss = .5 * tf.reduce_sum(tf.square(targets - values))

            # actor loss
            # use tf.where to get rid of -infinities, but probably it causes inability to calculate grads
            # check https://stackoverflow.com/questions/33712178/tensorflow-nan-bug/42497444#42497444
            # target_action_log_probs = tf.where(actions == -1, 0., target_action_log_probs)
            target_action_log_probs = target_action_log_probs * mask2d
            actor_loss = -1 * target_action_log_probs * td_error
            # actor_loss = tf.reduce_mean(actor_loss)
            actor_loss = tf.reduce_sum(actor_loss)

            # entropy loss
            entropy = tools.get_entropy_from_probs(probs, mask3d)
            # entropy_loss = -1 * self._entropy_c * tf.reduce_sum(entropy)
            # entropy_loss = -1 * self._entropy_c * tf.reduce_mean(entropy)
            foo = 1 - progress * (1 - self._entropy_c_decay)
            if self._is_debug:
                entropy_v = entropy.numpy()
                foo_v = foo.numpy()
            entropy_loss = -self._entropy_c * tf.reduce_sum(entropy * foo)

            loss = actor_loss + critic_loss + entropy_loss
        grads = tape.gradient(loss, self._model.trainable_variables)
        grads = [tf.clip_by_norm(g, 4.0) for g in grads]
        self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

        data_count = tf.reduce_sum(mask2d)
        return data_count

    def do_train(self):
        ds = tfrecords_storage.read_records_for_rl(self._feature_maps_shape, self._actions_shape,
                                                   "data/tfrecords/rl/")
        ds = ds.batch(self._batch_size)
        iterator = iter(ds)
        for step_counter in range(1, self._iterations_number + 1):

            # sampling
            sample = next(iterator)

            # training
            t1 = time.time()
            self._training_step(*sample)
            t2 = time.time()
            print(f"Training. Step: {step_counter} Time: {t2 - t1:.2f}.")

        print("RL training is done.")
