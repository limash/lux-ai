def pg_agent_run(config_in, data_in, current_cycle_in=None, global_var_actor_in=None):
    import abc
    import time
    import pickle
    import itertools
    import glob

    import numpy as np
    import tensorflow as tf
    import tensorflow_addons as tfa
    import ray
    # import reverb

    from lux_ai import models, tools, tfrecords_storage
    from lux_gym.envs.lux.action_vectors_new import empty_worker_action_vectors

    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    class Agent(abc.ABC):
        def __init__(self, config, data, current_cycle=None, global_var_actor=None):

            self._feature_maps_shape = tools.get_feature_maps_shape(config["environment"])
            self._actions_shape = [item.shape for item in empty_worker_action_vectors]
            self._model_name = config["model_name"]
            self._batch_size = config["batch_size"]
            if self._model_name == "actor_critic_residual_shrub":
                self._model = models.actor_critic_residual_shrub(self._actions_shape)
                self._model_actions_shape = self._actions_shape
            elif self._model_name == "actor_critic_residual_six_actions":
                self._model = models.actor_critic_residual_six_actions(6)
                self._model_actions_shape = 6
            else:
                raise NotImplementedError

            # launch a model once to define structure
            dummy_feature_maps = np.zeros(self._feature_maps_shape, dtype=np.float32)
            dummy_feature_maps[6, 6, :1] = 1
            dummy_input = tf.convert_to_tensor(dummy_feature_maps, dtype=tf.float32)
            dummy_input = tf.nest.map_structure(lambda x: tf.expand_dims(x, axis=0), dummy_input)
            self._model(dummy_input)
            # load weights
            files = glob.glob("./data/weights/*.pickle")
            files_n = len(files)
            if files_n > 0:
                with open(files[-1], 'rb') as file:
                    data = pickle.load(file)
                    print(f"Continue the model training from {files[-1]}.")
            elif data is not None:
                print("Continue the model training from data.pickle.")
            else:
                raise NotImplementedError
            self._model.set_weights(data['weights'])

            # self._n_points = config["n_points"]
            # self._iterations_number = config["iterations_number"]
            # self._save_interval = config["save_interval"]

            self._optimizer = tfa.optimizers.AdamW(weight_decay=1.e-5, learning_rate=config["default_lr"])
            self._entropy_c = config["entropy_c"]
            self._entropy_c_decay = config["entropy_c_decay"]
            # self._lambda = config["lambda"]

            self._is_debug = config["debug"]
            if not config["debug"]:
                self._training_step = tf.function(self._training_step)

            self._current_cycle = current_cycle
            self._global_var_actor = global_var_actor if global_var_actor else None

        def _training_step(self, actions, behaviour_policy_probs, observations, total_rewards,  progress):
            print("Tracing")

            if self._is_debug:
                actions_v = actions.numpy()
                behaviour_policy_probs_v = behaviour_policy_probs.numpy()
                observations_v = observations.numpy()
                total_rewards_v = total_rewards.numpy()
                progress_v = progress.numpy()

            behaviour_action_log_probs = tools.get_prob_logs_from_probs(behaviour_policy_probs, actions,
                                                                        self._model_actions_shape)
            if self._is_debug:
                behaviour_action_log_probs_v = behaviour_action_log_probs.numpy()

            with tf.GradientTape() as tape:
                maps = observations
                maps_shape = tf.shape(maps)
                # scalars_shape = tf.shape(scalars)
                maps_merged = tf.reshape(maps, (-1, maps_shape[2], maps_shape[3], maps_shape[4]))
                # maps_merged = tf.reshape(maps, (-1, maps_shape[2], maps_shape[3]))
                # scalars_merged = tf.reshape(scalars, (-1, scalars_shape[2]))
                probs_merged, values_merged = self._model(maps_merged, training=True)
                probs = tf.reshape(probs_merged, (maps_shape[0], maps_shape[1], -1))
                # values = tf.reshape(values_merged, (maps_shape[0], maps_shape[1], -1))
                # -

                # logits = tf.roll(logits, shift=1, axis=0)  # shift by 1 along time dimension, to match a pattern
                # values = tf.roll(values, shift=1, axis=0)  # where actions, logits, etc. led to the observation
                # probs = tf.roll(probs, shift=1, axis=1)  # shift by 1 along time dimension, to match a pattern
                # values = tf.roll(values, shift=1, axis=1)  # where actions, logits, etc. led to the observation
                target_action_log_probs = tools.get_prob_logs_from_probs(probs, actions, self._model_actions_shape)
                if self._is_debug:
                    probs_v = probs.numpy()
                    target_action_log_probs_v = target_action_log_probs.numpy()

                with tape.stop_recording():
                    log_rhos = target_action_log_probs - behaviour_action_log_probs
                    rhos = tf.exp(log_rhos)
                    # rhos_masked = tf.where(actions == -1, 0., rhos)  # use where to remove nans
                    rhos_masked = rhos * mask2d
                    clipped_rhos = tf.minimum(tf.constant(1.), rhos_masked)

                # add final rewards to 'empty' spots in values
                # values = tf.squeeze(values) * mask2d  # to ensure zeros in not valid spots
                # values = tf.where(e_mask == 0, total_rewards, values)  # to calculate targets
                # values = tf.where(mask2d == 0, total_rewards, tf.squeeze(values))  # to calculate targets
                if self._is_debug:
                    clipped_rhos_v = clipped_rhos.numpy()
                    # values_v = values.numpy()

                with tape.stop_recording():
                    # calculate targets
                    # targets = tools.prepare_td_lambda(tf.squeeze(values), returns, None, self._lambda, 1.)
                    # targets = tools.tf_prepare_td_lambda_no_rewards(values, total_rewards[:, 0], self._lambda, 1.)
                    # targets = targets * mask2d
                    targets = total_rewards * mask2d

                # values = values * mask2d
                if self._is_debug:
                    # values_v = values.numpy()
                    targets_v = targets.numpy()

                with tape.stop_recording():
                    # td error with truncated IS weights (rhos), it is a constant:
                    # modified_rhos = tf.math.divide_no_nan(1., clipped_rhos)
                    # modified_rhos = tf.minimum(tf.constant(2.), modified_rhos)
                    # td_error = modified_rhos * targets
                    # td_error = modified_rhos * (targets - values)
                    # td_error = clipped_rhos * (targets - values)
                    td_error = clipped_rhos * targets * loss_weights

                # critic loss
                # critic_loss = self._loss_fn(targets, values)
                # critic_loss = .5 * tf.reduce_sum(tf.square(targets - values))

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

                loss = actor_loss + entropy_loss  # + critic_loss
            grads = tape.gradient(loss, self._model.trainable_variables)
            # grads = [tf.clip_by_norm(g, 4.0) for g in grads]
            self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

            data_count = tf.reduce_sum(mask2d)
            return data_count

        def do_train(self):
            if self._current_cycle is not None:
                save_path = f'data/weights/{self._current_cycle}.pickle'
            else:
                save_path = f'data/weights/data.pickle'

            ds_learn = tfrecords_storage.read_records_for_rl_pg(
                self._feature_maps_shape, self._actions_shape, self._model_name,
                "data/tfrecords/rl/storage_0/"
            )
            ds_storage = tfrecords_storage.read_records_for_rl_pg(
                self._feature_maps_shape, self._actions_shape, self._model_name,
                "data/tfrecords/rl/storage/"
            )
            ds_learn = ds_learn.batch(self._batch_size).prefetch(1)
            ds_storage = ds_storage.batch(self._batch_size).prefetch(1)

            storage_iterator = iter(ds_storage)
            learn_iterator = iter(ds_learn)
            # for step_counter in range(1, self._iterations_number + 1):
            for step_counter in itertools.count(1):
                # sampling
                if step_counter % 3 == 0:
                    sample = next(storage_iterator)
                else:
                    try:
                        sample = next(learn_iterator)
                    except StopIteration:
                        break

                # training
                t1 = time.time()
                self._training_step(*sample)
                t2 = time.time()
                if step_counter % 100 == 0:
                    print(f"Training. Step: {step_counter} Time: {t2 - t1:.2f}.")

            weights = self._model.get_weights()
            data = {
                'weights': weights,
            }
            with open(save_path, 'wb') as f:
                pickle.dump(data, f, protocol=4)

            if self._global_var_actor is not None:
                ray.get(self._global_var_actor.set_done.remote(True))

            print("RL training is done.")

    ac_agent = Agent(config_in, data_in, current_cycle_in, global_var_actor_in)
    ac_agent.do_train()
