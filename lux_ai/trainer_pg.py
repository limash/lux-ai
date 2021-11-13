def pg_agent_run(config_in, data_in, global_var_actor_in=None, filenames_in=None, current_cycle_in=None):
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
        def __init__(self, config, data, global_var_actor=None, filenames=None, current_cycle=None):

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
            self._global_var_actor = global_var_actor
            self._filenames = filenames

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
                probs, values = self._model(observations, training=True)
                target_action_log_probs = tools.get_prob_logs_from_probs(probs, actions, self._model_actions_shape)
                if self._is_debug:
                    probs_v = probs.numpy()
                    target_action_log_probs_v = target_action_log_probs.numpy()

                with tape.stop_recording():
                    log_rhos = target_action_log_probs - behaviour_action_log_probs
                    rhos = tf.exp(log_rhos)
                    clipped_rhos = tf.minimum(tf.constant(1.), rhos)

                if self._is_debug:
                    clipped_rhos_v = clipped_rhos.numpy()

                targets = total_rewards
                with tape.stop_recording():
                    td_error = clipped_rhos * targets

                if self._is_debug:
                    td_error_v = td_error.numpy()

                # actor loss
                actor_loss = -1 * target_action_log_probs * td_error
                actor_loss = tf.reduce_sum(actor_loss)

                # entropy loss
                entropy = tools.get_entropy_from_probs(probs)
                entropy_loss = -1 * self._entropy_c * tf.reduce_sum(entropy)
                # entropy_loss = -1 * self._entropy_c * tf.reduce_mean(entropy)
                # foo = 1 - progress * (1 - self._entropy_c_decay)
                if self._is_debug:
                    entropy_v = entropy.numpy()
                    # foo_v = foo.numpy()
                # entropy_loss = -self._entropy_c * tf.reduce_sum(entropy * foo)

                loss = actor_loss + entropy_loss
            grads = tape.gradient(loss, self._model.trainable_variables)
            # grads = [tf.clip_by_norm(g, 4.0) for g in grads]
            self._optimizer.apply_gradients(zip(grads, self._model.trainable_variables))

        def do_train(self):
            if self._current_cycle is not None:
                save_path = f'data/weights/{self._current_cycle}.pickle'
            else:
                save_path = f'data/weights/data.pickle'

            ds_learn = tfrecords_storage.read_records_for_rl_pg(
                self._feature_maps_shape, self._actions_shape, self._model_name,
                "_",
                filenames=self._filenames
            )
            ds_learn = ds_learn.batch(self._batch_size).prefetch(1)
            learn_iterator = iter(ds_learn)

            for step_counter in itertools.count(1):
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
                    if self._global_var_actor is not None:
                        is_done = ray.get(self._global_var_actor.get_done.remote())
                        if is_done:
                            break

            weights = self._model.get_weights()
            data = {
                'weights': weights,
            }
            with open(save_path, 'wb') as f:
                pickle.dump(data, f, protocol=4)

            print("RL training is done.")

    ac_agent = Agent(config_in, data_in, global_var_actor_in, filenames_in, current_cycle_in)
    ac_agent.do_train()
