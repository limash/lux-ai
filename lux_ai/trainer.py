import abc
import pickle

import numpy as np
import tensorflow as tf
# import reverb

from lux_ai import models, tools, tfrecords_storage
from lux_gym.envs.lux.action_vectors import actions_number  # , worker_action_mask

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


class Agent(abc.ABC):

    def __init__(self, config, data):

        # self._entropy_c = config["entropy_c"]
        # self._entropy_c_decay = config["entropy_c_decay"]
        # self._lambda = config["lambda"]

        self._feature_maps_shape = tools.get_feature_maps_shape(config["environment"])
        self._actions_shape = actions_number
        if config["model_name"] == "actor_critic_custom":
            self._model = models.actor_critic_base(self._actions_shape)
            # launch a model once to define structure
            dummy_feature_maps = np.zeros(self._feature_maps_shape, dtype=np.float32)
            dummy_feature_maps[16, 16, :1] = 1
            # dummy_input = (tf.convert_to_tensor(dummy_feature_maps, dtype=tf.float32),
            #                tf.convert_to_tensor(worker_action_mask, dtype=tf.float32))
            dummy_input = tf.convert_to_tensor(dummy_feature_maps, dtype=tf.float32)
            dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
            self._model(dummy_input)
        else:
            raise ValueError

        class_weights = np.ones(self._actions_shape, dtype=np.single)
        class_weights[4] = 0.05
        # class_weights[5] = 2.
        class_weights = tf.convert_to_tensor(class_weights, dtype=tf.float32)
        self._class_weights = tf.expand_dims(class_weights, axis=0)
        self._loss_function = tools.skewed_kldivergence_loss(self._class_weights)

        if data is not None:
            self._model.set_weights(data['weights'])
            print("Continue the model training.")

        # self._is_debug = config["debug"]
        # if not config["debug"]:
        #     self._training_step_full = tf.function(self._training_step_full)
        # self._dataset = reverb.TrajectoryDataset.from_table_signature(
        #     server_address=f'localhost:{buffer_server_port}',
        #     table=buffer_table_names[0],
        #     max_in_flight_samples_per_worker=2*config["batch_size"])
        # self._client = reverb.Client(f'localhost:{buffer_server_port}')
        self._batch_size = config["batch_size"]

    def imitate(self):
        # for sample in self._dataset.take(1):
        #     actions = sample.data['actions']
        #     actions_v = actions.numpy()
        #     actions_probs = sample.data['actions_probs']
        #     actions_probs_v = actions_probs.numpy()
        #     actions_masks = sample.data['actions_masks']
        #     actions_masks_v = actions_masks.numpy()
        #     observations = sample.data['observations']
        #     observations_v = observations.numpy()
        #     total_rewards = sample.data['total_rewards']
        #     total_rewards_v = total_rewards.numpy()
        #     temporal_masks = sample.data['temporal_masks']
        #     temporal_masks_v = temporal_masks.numpy()
        #     progresses = sample.data['progresses']
        #     progresses_v = progresses.numpy()
        # info = self._client.server_info()
        # imitate_dataset = self._dataset.map(lambda x: ((x.data['observations'], x.data['actions_masks']),
        #                                                (x.data['actions_probs'], x.data['total_rewards'])
        #                                                )
        #                                     )
        # imitate_dataset = self._dataset.map(lambda x: ((tf.cast(x.data['observations'], dtype=tf.float32),
        #                                                 tf.cast(x.data['actions_masks'], dtype=tf.float32)),
        #                                                (tf.cast(x.data['actions_probs'], dtype=tf.float32),
        #                                                 tf.cast(x.data['total_rewards'], dtype=tf.float32))
        #                                                )
        #                                     )
        # batched_dataset = imitate_dataset.batch(self._batch_size, drop_remainder=True)
        # dataset = batched_dataset.map(tools.merge_first_two_dimensions)
        # raw_dataset = tf.data.TFRecordDataset('/home/shamilyakubov/src/lux-ai/'
        #                                       'data/tfrecords/imitator/train/26691017_Toad Brigade.tfrec')
        # features = {
        #     "observation": tf.io.FixedLenFeature([], tf.string),
        #     "action_mask": tf.io.FixedLenFeature([], tf.string),
        #     "action_probs": tf.io.FixedLenFeature([], tf.string),
        #     "reward": tf.io.FixedLenFeature([], tf.float32),
        # }
        # for raw_record in raw_dataset.take(1):
        #     raw_example = tf.train.Example()
        #     raw_example.ParseFromString(raw_record.numpy())
        #     print(raw_example)
        #     example = tf.io.parse_single_example(raw_record, features)
        #     observation = tf.io.parse_tensor(example["observation"], tf.string)
        #     observation = tf.expand_dims(observation, axis=0)
        #     observation = tf.io.deserialize_many_sparse(observation, dtype=tf.float16)
        #     observation = tf.sparse.to_dense(observation)
        #     observation = tf.squeeze(observation)
        #     print("Trololo")

        ds_train = tfrecords_storage.read_records_for_imitator(self._feature_maps_shape, self._actions_shape,
                                                               "data/tfrecords/imitator/train/")
        ds_valid = tfrecords_storage.read_records_for_imitator(self._feature_maps_shape, self._actions_shape,
                                                               "data/tfrecords/imitator/validation/")
        # ds_train = ds_train.map(lambda x1, x2, x3: (tf.cast(x1, dtype=tf.float32),
        #                                             (tf.cast(x2, dtype=tf.float32),
        #                                              tf.cast(x3, dtype=tf.float32))
        #                                             )
        #                         )
        # ds_valid = ds_valid.map(lambda x1, x2, x3: (tf.cast(x1, dtype=tf.float32),
        #                                             (tf.cast(x2, dtype=tf.float32),
        #                                              tf.cast(x3, dtype=tf.float32))
        #                                             )
        #                         )
        ds_train = ds_train.batch(self._batch_size)  # , drop_remainder=True)
        ds_valid = ds_valid.batch(self._batch_size)  # , drop_remainder=True)

        # for sample in ds_valid.take(10):
        #     # tfrecords_storage.random_reverse(sample)
        #     observations = sample[0].numpy()
        #     # observations = sample[0][0].numpy()
        #     # actions_masks = sample[0][1].numpy()
        #     actions_probs = sample[1][0].numpy()
        #     total_rewards = sample[1][1].numpy()
        #     probs_output, value_output = self._model(observations)
        #     probs_output_v = probs_output.numpy()
        #     value_output_v = value_output.numpy()
        #     skewed_loss = self._loss_function(sample[1][0], probs_output)
        #     loss = tf.keras.losses.kl_divergence(sample[1][0], probs_output)

        lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True,
        )

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),  # , clipnorm=4.),
            loss={
                "output_1": self._loss_function,  # tf.keras.losses.KLDivergence(),
                "output_2": None  # tf.keras.losses.MeanSquaredError()
            },
            metrics={
                "output_1": [tf.keras.metrics.CategoricalAccuracy()],
                # "value_output": [tf.keras.metrics.MeanAbsolutePercentageError(),
                #                  tf.keras.metrics.MeanAbsoluteError()]
            },
            # loss_weights={"output_1": 2.0}  # ,
            # "output_2": 0.1},
        )

        self._model.fit(ds_train, epochs=20, validation_data=ds_valid, callbacks=[early_stop_callback, lr_scheduler])
        weights = self._model.get_weights()
        data = {
            'weights': weights,
        }
        with open(f'data/data.pickle', 'wb') as f:
            pickle.dump(data, f, protocol=4)

        print("imitation done")

    # def _training_step_full(self, actions, behaviour_policy_logits, observations, rewards, dones,
    #                         total_rewards, progress, steps, info):
    #     print("Tracing")

    #     if self._is_debug:
    #         actions_v = actions.numpy()
    #         rewards_v = rewards.numpy()
    #         dones_v = dones.numpy()
    #         total_rewards_v = total_rewards.numpy()
    #         progress_v = progress.numpy()

    #     # actions = tf.transpose(actions)
    #     # behaviour_policy_logits = tf.transpose(behaviour_policy_logits, perm=[1, 0, 2])
    #     # maps = tf.transpose(observations[0], perm=[1, 0, 2, 3, 4])
    #     # scalars = tf.transpose(observations[1], perm=[1, 0, 2])
    #     # rewards = tf.transpose(rewards)
    #     # dones = tf.transpose(dones)

    #     # nsteps = tf.argmax(dones, axis=0, output_type=tf.int32)
    #     # ta = tf.TensorArray(dtype=tf.float32, size=self._sample_batch_size, dynamic_size=False)
    #     # for i in tf.range(self._sample_batch_size):
    #     #     row = tf.concat([tf.constant([0.]),
    #     #                      tf.linspace(0., 1., nsteps[i] + 1)[:-1],
    #     #                      tf.ones(steps - nsteps[i] - 1)], axis=0)
    #     #     ta = ta.write(i, row)
    #     # progress = ta.stack()
    #     # progress = tf.transpose(progress)

    #     # prepare a mask for the valid time steps
    #     # alive_positions = tf.where(actions != -1)
    #     # ones_array = tf.ones(alive_positions.shape[0])
    #     # mask = tf.scatter_nd(alive_positions, ones_array, actions.shape)
    #     # mask2d = tf.where(actions == -1, 0., 1.)
    #     mask2d = tf.concat((tf.zeros([tf.shape(dones)[0], 1]), (tf.ones_like(dones) - dones)[:, :-1]), axis=1)
    #     # e_mask = tf.concat((tf.zeros([tf.shape(episode_dones)[0], 1]),
    #     #                     (tf.ones_like(episode_dones) - episode_dones)[:, :-1]), axis=1)
    #     # mask3d = tf.where(behaviour_policy_logits == 0., 0., 1.)
    #     mask3d = tf.transpose(tf.ones([4, 1, 1]) * mask2d, perm=[1, 2, 0])
    #     if self._is_debug:
    #         mask2d_v = mask2d.numpy()
    #         mask3d_v = mask3d.numpy()
    #     # e_mask_v = e_mask.numpy()

    #     # get final rewards, currently there is the only reward in the end of a game
    #     # returns = total_rewards[-1, :]

    #     # behaviour_action_log_probs = -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=behaviour_policy_logits,
    #     #                                                                              labels=actions)
    #     # it is almost similar to above line, but above probably won't work on cpus (due to -1 actions)
    #     behaviour_action_log_probs = misc.get_prob_logs_from_logits(behaviour_policy_logits, actions,
    #                                                                 self._n_outputs)

    #     with tf.GradientTape() as tape:
    #         maps, scalars = observations
    #         if self._is_debug:
    #             maps_v = maps.numpy()
    #             scalars_v = scalars.numpy()
    #         # there are two ways to get outputs from the model
    #         # 1: using map_fn along the time dimension (or whatever), it is slow but consumes less memory
    #         # logits, values = tf.map_fn(self._model, (maps, scalars),
    #         #                            fn_output_signature=[tf.TensorSpec((self._sample_batch_size,
    #         #                                                                self._n_outputs), dtype=tf.float32),
    #         #                                                 tf.TensorSpec((self._sample_batch_size, 1),
    #         #                                                               dtype=tf.float32)])
    #         # -
    #         # 2: merging time and batch dimensions and applying the model at once, it is fast, but requires gpu memory
    #         maps_shape = tf.shape(maps)
    #         scalars_shape = tf.shape(scalars)
    #         maps_merged = tf.reshape(maps, (-1, maps_shape[2], maps_shape[3], maps_shape[4]))
    #         # maps_merged = tf.reshape(maps, (-1, maps_shape[2], maps_shape[3]))
    #         scalars_merged = tf.reshape(scalars, (-1, scalars_shape[2]))
    #         logits_merged, values_merged = self._model((maps_merged, scalars_merged), training=True)
    #         logits = tf.reshape(logits_merged, (scalars_shape[0], scalars_shape[1], -1))
    #         values = tf.reshape(values_merged, (scalars_shape[0], scalars_shape[1], -1))
    #         # -

    #         # logits = tf.roll(logits, shift=1, axis=0)  # shift by 1 along time dimension, to match a pattern
    #         # values = tf.roll(values, shift=1, axis=0)  # where actions, logits, etc. led to the observation
    #         logits = tf.roll(logits, shift=1, axis=1)  # shift by 1 along time dimension, to match a pattern
    #         values = tf.roll(values, shift=1, axis=1)  # where actions, logits, etc. led to the observation
    #         target_action_log_probs = misc.get_prob_logs_from_logits(logits, actions, self._n_outputs)

    #         with tape.stop_recording():
    #             log_rhos = target_action_log_probs - behaviour_action_log_probs
    #             rhos = tf.exp(log_rhos)
    #             # rhos_masked = tf.where(actions == -1, 0., rhos)  # use where to remove nans, should be outside tape
    #             rhos_masked = rhos * mask2d
    #             clipped_rhos = tf.minimum(tf.constant(1.), rhos_masked)

    #         # add final rewards to 'empty' spots in values
    #         # values = tf.squeeze(values) * mask2d  # to ensure zeros in not valid spots
    #         # values = tf.where(e_mask == 0, total_rewards, values)  # to calculate targets
    #         values = tf.where(mask2d == 0, total_rewards, tf.squeeze(values))  # to calculate targets
    #         if self._is_debug:
    #             clipped_rhos_v = clipped_rhos.numpy()
    #             values_v = values.numpy()

    #         with tape.stop_recording():
    #             # calculate targets
    #             # targets = misc.prepare_td_lambda(tf.squeeze(values), returns, None, self._lambda, 1.)
    #             targets = misc.tf_prepare_td_lambda_no_rewards(values, total_rewards[:, -1], self._lambda, 1.)
    #             targets = targets * mask2d

    #         values = values * mask2d
    #         if self._is_debug:
    #             values_v = values.numpy()
    #             targets_v = targets.numpy()

    #         with tape.stop_recording():
    #             # td error with truncated IS weights (rhos), it is a constant:
    #             td_error = clipped_rhos * (targets - values)

    #         # critic loss
    #         # critic_loss = self._loss_fn(targets, values)
    #         critic_loss = .5 * tf.reduce_sum(tf.square(targets - values))

    #         # actor loss
    #         # use tf.where to get rid of -infinities, but probably it causes inability to calculate grads
    #         # check https://stackoverflow.com/questions/33712178/tensorflow-nan-bug/42497444#42497444
    #         # target_action_log_probs = tf.where(actions == -1, 0., target_action_log_probs)
    #         target_action_log_probs = target_action_log_probs * mask2d
    #         actor_loss = -1 * target_action_log_probs * td_error
    #         # actor_loss = tf.reduce_mean(actor_loss)
    #         actor_loss = tf.reduce_sum(actor_loss)

    #         # entropy loss
    #         entropy = misc.get_entropy(logits, mask3d)
    #         # entropy_loss = -1 * self._entropy_c * tf.reduce_sum(entropy)
    #         # entropy_loss = -1 * self._entropy_c * tf.reduce_mean(entropy)
    #         foo = 1 - progress * (1 - self._entropy_c_decay)
    #         if self._is_debug:
    #             entropy_v = entropy.numpy()
    #             foo_v = foo.numpy()
    #         entropy_loss = -self._entropy_c * tf.reduce_sum(entropy * foo)

    #         loss = actor_loss + critic_loss + entropy_loss
    #     grads = tape.gradient(loss, self._model.trainable_variables)
    #     grads = [tf.clip_by_norm(g, 4.0) for g in grads]
    #     self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

    #     data_count = tf.reduce_sum(mask2d)
    #     return data_count

    # def _train(self, samples_in):
    #     for i, sample in enumerate(samples_in):
    #         if sample is not None:
    #             # passing i to tf function as python variable cause the same number of retraces
    #             # as passing it as a tf constant
    #             # i = tf.constant(i, dtype=tf.float32)
    #             # also passing a tuple of tf constants does not cause retracing
    #             if self._is_policy_gradient:
    #                 if self._is_full_episode:
    #                     data_count = self._training_step_full(*sample.data, self._n_points, sample.info)
    #                     return data_count
    #                 else:
    #                     self._training_step(*sample.data, i + 2, info=info)
    #             else:
    #                 action, obs, reward, done = sample.data
    #                 key, probability, table_size, priority = sample.info
    #                 experiences, info = (action, obs, reward, done), (key, probability, table_size, priority)
    #                 if self._is_full_episode:
    #                     self._training_step_full(*experiences, steps=self._n_points, info=info)
    #                 else:
    #                     self._training_step(*experiences, steps=i + 2, info=info)

    # def do_train(self, iterations_number=20000, save_interval=2000):

    #     target_model_update_interval = 3000
    #     epsilon_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    #         initial_learning_rate=self._start_epsilon,
    #         decay_steps=iterations_number,
    #         end_learning_rate=self._final_epsilon) if self._start_epsilon is not None else None

    #     weights = None
    #     mask = None
    #     # rewards = 0
    #     # steps = 0
    #     print_interval = 100
    #     update_interval = print_interval
    #     # eval_counter = 0
    #     data_counter = 0

    #     lr = self._default_lr * self._data_cnt_ema
    #     self._optimizer.learning_rate.assign(lr)

    #     # wait if there are not enough data in the buffer
    #     while True:
    #         items_created = []
    #         for table_name in self._table_names:
    #             server_info = self._replay_memory_client.server_info()[table_name]
    #             items_total = server_info.current_size + server_info.num_deleted_episodes
    #             items_created.append(items_total)

    #         if items_created[-1] < self._sample_batch_size:
    #             print("Waiting to collect enough data.")
    #             time.sleep(1)
    #             continue
    #         else:
    #             break

    #     weights = self._model.get_weights()
    #     # print(f" Variables: {len(self._model.trainable_variables)}")
    #     ray.get(self._workers_info.set_current_weights.remote((weights, 0)))

    #     # the main training loop
    #     for step_counter in range(1, iterations_number + 1):

    #         # sampling
    #         samples = self._sample_experience()

    #         # training
    #         # t1 = time.time()
    #         data_count = self._train(samples)
    #         data_counter += data_count.numpy()
    #         # t2 = time.time()
    #         # print(f"Training. Step: {step_counter} Time: {t2 - t1}")

    #         if step_counter % update_interval == 0:
    #             if not self._ray_queue.full():
    #                 weights = self._model.get_weights()
    #                 self._ray_queue.put(weights)  # send weights to the interprocess ray queue
    #                 # print("Put weights in queue.")

    #         if step_counter % print_interval == 0:
    #             lr = self._get_learning_rate(data_counter, print_interval, step_counter)
    #             self._optimizer.learning_rate.assign(lr)
    #             # lr = self._optimizer.learning_rate.numpy()
    #             data_counter = 0

    #             items_prev = items_created
    #             # get from a buffer the total number of created elements since a buffer initialization
    #             items_created = []
    #             for table_name in self._table_names:
    #                 server_info = self._replay_memory_client.server_info()[table_name]
    #                 items_total = server_info.current_size + server_info.num_deleted_episodes
    #                 items_created.append(items_total)

    #             # fraction = [x / y if x != 0 else 1.e-9 for x, y in zip(self._items_sampled, items_created)]
    #             per_step_items_created = items_created[-1] - items_prev[-1]
    #             if per_step_items_created == 0:
    #                 step_fraction = self._sample_batch_size * print_interval
    #             else:
    #                 step_fraction = self._sample_batch_size * print_interval / per_step_items_created

    #             print(f"Step: {step_counter}, Sampled: {self._items_sampled[0]}, "
    #                   f"Created total: {items_created[0]}, "
    #                   f"Step sample/creation frac: {step_fraction:.2f}, "
    #                   f"LR: {lr:.2e}")

    #         # evaluation
    #         if step_counter % save_interval == 0:
    #             # eval_counter += 1
    #             # epsilon = 0 if epsilon_fn is not None else None
    #             # mean_episode_reward, mean_steps = self._evaluate_episodes(epsilon=epsilon)
    #             # print("----Evaluation------------------")
    #             # print(f"Iteration:{step_counter:.2f}; "
    #             #       f"Reward: {mean_episode_reward:.2f}; "
    #             #       f"Steps: {mean_steps:.2f}")
    #             # print("--------------------------------")
    #             # rewards += mean_episode_reward
    #             # steps += mean_steps

    #             weights = self._model.get_weights()
    #             ray.get(self._workers_info.set_current_weights.remote((weights, step_counter)))
    #             data = {
    #                 'weights': weights,
    #             }
    #             with open(f'data/data{step_counter}.pickle', 'wb') as f:
    #                 pickle.dump(data, f, protocol=4)

    #             # with open('data/checkpoint', 'w') as text_file:
    #             #     checkpoint = self._replay_memory_client.checkpoint()
    #             #     print(checkpoint, file=text_file)

    #         # update target model weights
    #         if self._target_model and step_counter % target_model_update_interval == 0:
    #             weights = self._model.get_weights()
    #             self._target_model.set_weights(weights)

    #         # store weights at the last step
    #         if step_counter % iterations_number == 0:
    #             # print("----Final-results---------------")
    #             # epsilon = 0 if epsilon_fn is not None else None
    #             # mean_episode_reward, mean_steps = self._evaluate_episodes(num_episodes=10, epsilon=epsilon)
    #             # print(f"Final reward with a model policy is {mean_episode_reward:.2f}; "
    #             #       f"Final average steps survived is {mean_steps:.2f}")
    #             # output_reward = rewards / eval_counter
    #             # output_steps = steps / eval_counter
    #             # print(f"Average episode reward with a model policy is {output_reward:.2f}; "
    #             #       f"Final average per episode steps survived is {output_steps:.2f}")
    #             # print("--------------------------------")

    #             weights = self._model.get_weights()
    #             mask = list(map(lambda x: np.where(np.abs(x) < 0.1, 0., 1.), weights))

    #             if self._make_checkpoint:
    #                 try:
    #                     checkpoint = self._replay_memory_client.checkpoint()
    #                 except RuntimeError as err:
    #                     print(err)
    #                     checkpoint = err
    #             else:
    #                 checkpoint = None

    #             # disable collectors
    #             if self._workers_info is not None:
    #                 ray.get(self._workers_info.set_done.remote(True))

    #             break

    #     return weights, mask, checkpoint
