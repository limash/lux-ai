# move all imports inside functions to use ray.remote multitasking


def actor_critic_residual(actions_shape):
    import tensorflow as tf
    import tensorflow.keras as keras

    class ResidualUnit(keras.layers.Layer):
        def __init__(self, filters, initializer, activation, **kwargs):
            super().__init__(**kwargs)

            self._filters = filters
            self._activation = activation
            self._conv = keras.layers.Conv2D(filters, 3, kernel_initializer=initializer, padding="same", use_bias=False)
            self._norm = keras.layers.BatchNormalization()

        def call(self, inputs, training=False, **kwargs):
            x = self._conv(inputs)
            x = self._norm(x, training=training)
            return self._activation(inputs + x)

        def compute_output_shape(self, batch_input_shape):
            batch, x, y, _ = batch_input_shape
            return [batch, x, y, self._filters]

    class ResidualModel(keras.Model):
        def __init__(self, actions_number, **kwargs):
            super().__init__(**kwargs)

            filters = 200
            layers = 10

            initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
            activation = keras.activations.relu

            self._conv = keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, use_bias=False)
            self._norm = keras.layers.BatchNormalization()
            self._activation = keras.layers.ReLU()
            self._residual_block = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]

            # self._depthwise = keras.layers.DepthwiseConv2D(32)
            self._depthwise = keras.layers.DepthwiseConv2D(13)
            self._flatten = keras.layers.Flatten()

            # self._city_tiles_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
            # self._city_tiles_probs1 = keras.layers.Dense(4, activation="softmax",
            #                                              kernel_initializer=initializer_random)
            self._workers_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
            self._workers_probs1 = keras.layers.Dense(actions_number, activation="softmax",
                                                      kernel_initializer=initializer_random)
            # self._carts_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
            # self._carts_probs1 = keras.layers.Dense(17, activation="softmax", kernel_initializer=initializer_random)

            self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                                activation=keras.activations.tanh)

        def call(self, inputs, training=False, mask=None):
            features = inputs
            x = features

            x = self._conv(x)
            x = self._norm(x, training=training)
            x = self._activation(x)

            for layer in self._residual_block:
                x = layer(x, training=training)

            shape_x = tf.shape(x)
            y = tf.reshape(x, (shape_x[0], -1, shape_x[-1]))
            y = tf.reduce_mean(y, axis=1)

            z1 = (x * features[:, :, :, :1])
            shape_z = tf.shape(z1)
            z1 = tf.reshape(z1, (shape_z[0], -1, shape_z[-1]))
            z1 = tf.reduce_sum(z1, axis=1)
            z2 = self._depthwise(x)
            z2 = self._flatten(z2)
            z = tf.concat([z1, z2], axis=1)

            # t = self._city_tiles_probs0(z)
            # t = self._city_tiles_probs1(t)
            w = self._workers_probs0(z)
            w = self._workers_probs1(w)
            # c = self._carts_probs0(z)
            # c = self._carts_probs1(c)
            # probs = tf.concat([t, w, c], axis=1)
            # probs = probs * actions_mask
            probs = w

            baseline = self._baseline(tf.concat([y, z], axis=1))

            return probs, baseline

        def get_config(self):
            pass

    model = ResidualModel(actions_shape)
    return model


def actor_critic_efficient(actions_shape):
    import copy
    import itertools

    import tensorflow as tf
    import tensorflow.keras as keras

    # import lux_ai.effnetv2_model as eff_model
    import lux_ai.hparams as hparams
    import lux_ai.effnetv2_configs as effnetv2_configs
    import lux_ai.utils as utils
    from lux_ai.effnetv2_model import round_filters, round_repeats, Stem, MBConvBlock, FusedMBConvBlock

    class EfficientModel(keras.Model):
        def __init__(self, actions_number, **kwargs):
            model_name = 'efficientnetv2-s'
            super().__init__(name=model_name, **kwargs)

            cfg = copy.deepcopy(hparams.base_config)
            if model_name:
                cfg.override(effnetv2_configs.get_model_config(model_name))
            self.cfg = cfg
            self._mconfig = cfg.model

            self._stem = Stem(self._mconfig, self._mconfig.blocks_args[0].input_filters)

            self._blocks = []
            block_id = itertools.count(0)
            block_name = lambda: 'blocks_%d' % next(block_id)
            for block_args in self._mconfig.blocks_args:
                assert block_args.num_repeat > 0
                # Update block input and output filters based on depth multiplier.
                input_filters = round_filters(block_args.input_filters, self._mconfig)
                output_filters = round_filters(block_args.output_filters, self._mconfig)
                repeats = round_repeats(block_args.num_repeat,
                                        self._mconfig.depth_coefficient)
                block_args.update(
                    dict(
                        input_filters=input_filters,
                        output_filters=output_filters,
                        num_repeat=repeats))

                # The first block needs to take care of stride and filter size increase.
                conv_block = {0: MBConvBlock, 1: FusedMBConvBlock}[block_args.conv_type]
                self._blocks.append(
                    conv_block(block_args, self._mconfig, name=block_name()))
                if block_args.num_repeat > 1:  # rest of blocks with the same block_arg
                    # pylint: disable=protected-access
                    block_args.input_filters = block_args.output_filters
                    block_args.strides = 1
                    # pylint: enable=protected-access
                for _ in range(block_args.num_repeat - 1):
                    self._blocks.append(
                        conv_block(block_args, self._mconfig, name=block_name()))

            initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
            activation = utils.get_act_fn(self._mconfig.act_fn)

            self._depthwise = keras.layers.DepthwiseConv2D(13)
            self._flatten = keras.layers.Flatten()

            self._workers_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
            self._workers_probs1_0 = keras.layers.Dense(actions_number[0][0], activation="softmax",
                                                        kernel_initializer=initializer_random)
            self._workers_probs1_1 = keras.layers.Dense(actions_number[1][0], activation="softmax",
                                                        kernel_initializer=initializer_random)
            self._workers_probs1_2 = keras.layers.Dense(actions_number[2][0], activation="softmax",
                                                        kernel_initializer=initializer_random)
            self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                                activation=keras.activations.tanh)

        def call(self, inputs, training=False, mask=None):
            outputs = self._stem(inputs, training)
            for idx, block in enumerate(self._blocks):
                # survival_prob = self._mconfig.survival_prob
                # if survival_prob:
                #     drop_rate = 1.0 - survival_prob
                #     survival_prob = 1.0 - drop_rate * float(idx) / len(self._blocks)
                survival_prob = 1.0
                outputs = block(outputs, training=training, survival_prob=survival_prob)

            x = outputs

            shape_x = tf.shape(x)
            y = tf.reshape(x, (shape_x[0], -1, shape_x[-1]))
            y = tf.reduce_mean(y, axis=1)

            z1 = (x * inputs[:, :, :, :1])
            shape_z = tf.shape(z1)
            z1 = tf.reshape(z1, (shape_z[0], -1, shape_z[-1]))
            z1 = tf.reduce_sum(z1, axis=1)
            z2 = self._depthwise(x)
            z2 = self._flatten(z2)
            z = tf.concat([z1, z2], axis=1)

            w = self._workers_probs0(z)
            probs1 = self._workers_probs1_0(w)
            probs2 = self._workers_probs1_1(w)
            probs3 = self._workers_probs1_2(w)
            # probs = (probs1, probs2, probs3)

            baseline = self._baseline(tf.concat([y, z], axis=1))

            return probs1, probs2, probs3, baseline

        def get_config(self):
            pass

    model = EfficientModel(actions_shape)
    # model = eff_model.get_model("efficientnetv2-s", weights=None)
    return model
