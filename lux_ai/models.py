# move all imports inside functions to use ray.remote multitasking


def actor_critic_residual_six_actions(actions_shape):
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


def actor_critic_sep_residual_six_actions(actions_shape):
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

    class CriticBranch(keras.layers.Layer):
        def __init__(self, filters, initializer, activation, layers, **kwargs):
            super().__init__(**kwargs)

            self._residual_block = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]
            self._depthwise = keras.layers.DepthwiseConv2D(13)
            self._flatten = keras.layers.Flatten()
            self._fc_128 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)

        def call(self, inputs, training=False, **kwargs):
            x, center = inputs

            for layer in self._residual_block:
                x = layer(x, training=training)

            shape_x = tf.shape(x)
            y = tf.reshape(x, (shape_x[0], -1, shape_x[-1]))
            y = tf.reduce_mean(y, axis=1)

            z1 = (x * center)
            shape_z = tf.shape(z1)
            z1 = tf.reshape(z1, (shape_z[0], -1, shape_z[-1]))
            z1 = tf.reduce_sum(z1, axis=1)
            z2 = self._depthwise(x)
            z2 = self._flatten(z2)
            z = tf.concat([y, z1, z2], axis=1)
            z = self._fc_128(z)

            return z

    class ActorBranch(keras.layers.Layer):
        def __init__(self, filters, initializer, activation, layers, **kwargs):
            super().__init__(**kwargs)

            self._residual_block = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]
            self._depthwise = keras.layers.DepthwiseConv2D(13)
            self._flatten = keras.layers.Flatten()
            self._fc_128 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)

        def call(self, inputs, training=False, **kwargs):
            x, center = inputs

            for layer in self._residual_block:
                x = layer(x, training=training)

            z1 = (x * center)
            shape_z = tf.shape(z1)
            z1 = tf.reshape(z1, (shape_z[0], -1, shape_z[-1]))
            z1 = tf.reduce_sum(z1, axis=1)
            z2 = self._depthwise(x)
            z2 = self._flatten(z2)
            z = tf.concat([z1, z2], axis=1)
            z = self._fc_128(z)
            return z

    class ResidualModel(keras.Model):
        def __init__(self, actions_number, **kwargs):
            super().__init__(**kwargs)

            initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
            activation = keras.activations.relu

            self._root = keras.layers.Conv2D(200, 3, padding="same", kernel_initializer=initializer, use_bias=False)
            self._root_norm = keras.layers.BatchNormalization()
            self._root_activation = keras.layers.ReLU()

            # actor
            actor_filters = 200
            actor_layers = 10
            self._actor_branch = ActorBranch(actor_filters, initializer, activation, actor_layers)
            self._action_type = keras.layers.Dense(actions_number, activation="softmax",
                                                   kernel_initializer=initializer_random)
            # critic
            critic_filters = 200
            critic_layers = 10
            self._critic_branch = CriticBranch(critic_filters, initializer, activation, critic_layers)
            self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                                activation=keras.activations.tanh)

        def call(self, inputs, training=False, mask=None):
            features = inputs
            x = features

            x = self._root(x)
            x = self._root_norm(x, training=training)
            x = self._root_activation(x)

            center = features[:, :, :, :1]
            z = (x, center)

            w1 = self._actor_branch(z, training=training)
            action_probs = self._action_type(w1)

            w2 = self._critic_branch(z, training=training)
            baseline = self._baseline(w2)

            return action_probs, baseline

        def get_config(self):
            pass

    model = ResidualModel(actions_shape)
    return model


def actor_critic_residual_with_transfer():
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

    class ActorBranch(keras.layers.Layer):
        def __init__(self, filters, initializer, activation, layers, **kwargs):
            super().__init__(**kwargs)

            self._residual_block = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]
            self._depthwise = keras.layers.DepthwiseConv2D(13)
            self._flatten = keras.layers.Flatten()
            self._fc_200 = keras.layers.Dense(200, activation=activation, kernel_initializer=initializer)

        def call(self, inputs, training=False, **kwargs):
            x, center = inputs

            for layer in self._residual_block:
                x = layer(x, training=training)

            z1 = (x * center)
            shape_z = tf.shape(z1)
            z1 = tf.reshape(z1, (shape_z[0], -1, shape_z[-1]))
            z1 = tf.reduce_sum(z1, axis=1)
            z2 = self._depthwise(x)
            z2 = self._flatten(z2)
            z = tf.concat([z1, z2], axis=1)
            z = self._fc_200(z)
            return z

    class ResidualModel(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            filters = 200
            stem_layers = 10
            branch_layers = 1

            initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
            activation = keras.activations.relu

            self._root = keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, use_bias=False)
            self._root_norm = keras.layers.BatchNormalization()
            self._root_activation = keras.layers.ReLU()
            self._stem = [ResidualUnit(filters, initializer, activation) for _ in range(stem_layers)]

            # action type: north, east, south, west, idle, build, transfer
            self._action_type_branch = ActorBranch(filters, initializer, activation, branch_layers)
            self._action_type = keras.layers.Dense(7, activation="softmax",
                                                   kernel_initializer=initializer_random)
            # transfer direction: north, east, south, west
            self._transfer_direction_branch = ActorBranch(filters, initializer, activation, branch_layers)
            self._transfer_direction = keras.layers.Dense(4, activation="softmax",
                                                          kernel_initializer=initializer_random)
            # resource to transfer: wood, coal, uranium
            self._transfer_resource_branch = ActorBranch(filters, initializer, activation, branch_layers)
            self._transfer_resource = keras.layers.Dense(3, activation="softmax",
                                                         kernel_initializer=initializer_random)

            self._depthwise = keras.layers.DepthwiseConv2D(13)
            self._flatten = keras.layers.Flatten()
            self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                                activation=keras.activations.tanh)

        def call(self, inputs, training=False, mask=None):
            features = inputs
            x = features

            x = self._root(x)
            x = self._root_norm(x, training=training)
            x = self._root_activation(x)

            for layer in self._stem:
                x = layer(x, training=training)

            center = features[:, :, :, :1]
            z = (x, center)

            w1 = self._action_type_branch(z, training=training)
            action_type_probs = self._action_type(w1)

            w3 = self._transfer_direction_branch(z, training=training)
            transfer_direction_probs = self._transfer_direction(w3)

            w4 = self._transfer_resource_branch(z, training=training)
            transfer_resource_probs = self._transfer_resource(w4)

            shape_x = tf.shape(x)
            y = tf.reshape(x, (shape_x[0], -1, shape_x[-1]))
            y = tf.reduce_mean(y, axis=1)

            o1 = (x * center)
            shape_o1 = tf.shape(o1)
            o1 = tf.reshape(o1, (shape_o1[0], -1, shape_o1[-1]))
            o1 = tf.reduce_sum(o1, axis=1)
            o2 = self._depthwise(x)
            o2 = self._flatten(o2)
            o = tf.concat([o1, o2], axis=1)

            baseline = self._baseline(tf.concat([y, o], axis=1))

            return action_type_probs, transfer_direction_probs, transfer_resource_probs, baseline

        def get_config(self):
            pass

    model = ResidualModel()
    return model


def actor_critic_residual_shrub():
    import tensorflow as tf
    import tensorflow.keras as keras

    class ResidualUnit(keras.layers.Layer):
        def __init__(self, filters, initializer, activation, **kwargs):
            super().__init__(**kwargs)

            self._filters = filters
            self._activation = activation
            self._conv = keras.layers.Conv2D(filters, 3, kernel_initializer=initializer,
                                             kernel_regularizer=keras.regularizers.l2(l2=1.e-5),
                                             padding="same", use_bias=False)
            self._norm = keras.layers.BatchNormalization()

        def call(self, inputs, training=False, **kwargs):
            x = self._conv(inputs)
            x = self._norm(x, training=training)
            return self._activation(inputs + x)

        def compute_output_shape(self, batch_input_shape):
            batch, x, y, _ = batch_input_shape
            return [batch, x, y, self._filters]

    class ActorBranch(keras.layers.Layer):
        def __init__(self, filters, initializer, activation, layers, **kwargs):
            super().__init__(**kwargs)

            self._residual_block = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]
            self._depthwise = keras.layers.DepthwiseConv2D(13)
            self._flatten = keras.layers.Flatten()
            self._fc = keras.layers.Dense(filters, activation=activation, kernel_initializer=initializer)

        def call(self, inputs, training=False, **kwargs):
            x, center = inputs

            for layer in self._residual_block:
                x = layer(x, training=training)

            z1 = (x * center)
            shape_z = tf.shape(z1)
            z1 = tf.reshape(z1, (shape_z[0], -1, shape_z[-1]))
            z1 = tf.reduce_sum(z1, axis=1)
            z2 = self._depthwise(x)
            z2 = self._flatten(z2)
            z = tf.concat([z1, z2], axis=1)
            z = self._fc(z)
            return z

    class ResidualModel(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            filters = 200
            stem_layers = 6
            branch_layers = 4

            initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
            activation = keras.activations.relu

            self._root = keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer,
                                             kernel_regularizer=keras.regularizers.l2(l2=1.e-5),
                                             use_bias=False)
            self._root_norm = keras.layers.BatchNormalization()
            self._root_activation = keras.layers.ReLU()

            # movement direction
            self._movement_direction_branch = ActorBranch(filters, initializer, activation, layers=10)
            self._movement_direction = keras.layers.Dense(4, activation="softmax",
                                                          kernel_initializer=initializer_random)

            self._stem = [ResidualUnit(filters, initializer, activation) for _ in range(stem_layers)]
            # build
            self._build_branch = ActorBranch(filters, initializer, activation, branch_layers)
            self._build = keras.layers.Dense(2, activation="softmax", kernel_initializer=initializer_random)
            # idle
            self._idle_branch = ActorBranch(filters, initializer, activation, branch_layers)
            self._idle = keras.layers.Dense(2, activation="softmax", kernel_initializer=initializer_random)
            # transfer
            self._transfer_branch = ActorBranch(filters, initializer, activation, branch_layers)
            self._transfer = keras.layers.Dense(2, activation="softmax", kernel_initializer=initializer_random)
            # transfer direction
            self._transfer_direction_branch = ActorBranch(filters, initializer, activation, branch_layers)
            self._transfer_direction = keras.layers.Dense(4, activation="softmax",
                                                          kernel_initializer=initializer_random)
            # resource to transfer
            self._transfer_resource_branch = ActorBranch(filters, initializer, activation, branch_layers)
            self._transfer_resource = keras.layers.Dense(3, activation="softmax",
                                                         kernel_initializer=initializer_random)

        def call(self, inputs, training=False, mask=None):
            features = inputs
            x = features

            x = self._root(x)
            x = self._root_norm(x, training=training)
            x = self._root_activation(x)

            center = features[:, :, :, :1]
            z1 = (x, center)

            w1 = self._movement_direction_branch(z1, training=training)
            movement_direction_probs = self._movement_direction(w1)

            for layer in self._stem:
                x = layer(x, training=training)
            z2 = (x, center)

            w2 = self._build_branch(z2, training=training)
            build_switch = self._build(w2)

            w3 = self._idle_branch(z2, training=training)
            idle_switch = self._idle(w3)

            w4 = self._transfer_branch(z2, training=training)
            transfer_switch = self._transfer(w4)

            w5 = self._transfer_direction_branch(z2, training=training)
            transfer_direction_probs = self._transfer_direction(w5)

            w6 = self._transfer_resource_branch(z2, training=training)
            transfer_resource_probs = self._transfer_resource(w6)

            return movement_direction_probs, build_switch, idle_switch, \
                   transfer_switch, transfer_direction_probs, transfer_resource_probs

        def get_config(self):
            pass

    model = ResidualModel()
    return model


def actor_critic_residual_switch_shrub():
    import tensorflow as tf
    import tensorflow.keras as keras

    class ResidualUnit(keras.layers.Layer):
        def __init__(self, filters, initializer, activation, **kwargs):
            super().__init__(**kwargs)

            self._filters = filters
            self._activation = activation
            self._conv = keras.layers.Conv2D(filters, 3, kernel_initializer=initializer,
                                             kernel_regularizer=keras.regularizers.l2(l2=1.e-5),
                                             padding="same", use_bias=False)
            self._norm = keras.layers.BatchNormalization()

        def call(self, inputs, training=False, **kwargs):
            x = self._conv(inputs)
            x = self._norm(x, training=training)
            return self._activation(inputs + x)

        def compute_output_shape(self, batch_input_shape):
            batch, x, y, _ = batch_input_shape
            return [batch, x, y, self._filters]

    class ActorBranch(keras.layers.Layer):
        def __init__(self, filters, initializer, activation, layers, **kwargs):
            super().__init__(**kwargs)

            self._residual_block = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]
            self._depthwise = keras.layers.DepthwiseConv2D(13)
            self._flatten = keras.layers.Flatten()
            self._fc = keras.layers.Dense(filters, activation=activation, kernel_initializer=initializer)

        def call(self, inputs, training=False, **kwargs):
            x, center = inputs

            for layer in self._residual_block:
                x = layer(x, training=training)

            z1 = (x * center)
            shape_z = tf.shape(z1)
            z1 = tf.reshape(z1, (shape_z[0], -1, shape_z[-1]))
            z1 = tf.reduce_sum(z1, axis=1)
            z2 = self._depthwise(x)
            z2 = self._flatten(z2)
            z = tf.concat([z1, z2], axis=1)
            z = self._fc(z)
            return z

    class ResidualModel(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            filters = 200
            stem_layers1 = 4
            branch_layers1 = 6
            stem_layers2 = 6
            branch_layers2 = 4

            initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
            activation = keras.activations.relu

            self._root = keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer,
                                             kernel_regularizer=keras.regularizers.l2(l2=1.e-5),
                                             use_bias=False)
            self._root_norm = keras.layers.BatchNormalization()
            self._root_activation = keras.layers.ReLU()

            # stem1
            self._stem1 = [ResidualUnit(filters, initializer, activation) for _ in range(stem_layers1)]
            # movement directions
            self._north_branch = ActorBranch(filters, initializer, activation, branch_layers1)
            self._north = keras.layers.Dense(2, activation="softmax", kernel_initializer=initializer_random)
            self._east_branch = ActorBranch(filters, initializer, activation, branch_layers1)
            self._east = keras.layers.Dense(2, activation="softmax", kernel_initializer=initializer_random)
            self._south_branch = ActorBranch(filters, initializer, activation, branch_layers1)
            self._south = keras.layers.Dense(2, activation="softmax", kernel_initializer=initializer_random)
            self._west_branch = ActorBranch(filters, initializer, activation, branch_layers1)
            self._west = keras.layers.Dense(2, activation="softmax", kernel_initializer=initializer_random)
            # build
            self._build_branch = ActorBranch(filters, initializer, activation, branch_layers1)
            self._build = keras.layers.Dense(2, activation="softmax", kernel_initializer=initializer_random)
            # idle
            self._idle_branch = ActorBranch(filters, initializer, activation, branch_layers1)
            self._idle = keras.layers.Dense(2, activation="softmax", kernel_initializer=initializer_random)

            # stem2
            self._stem2 = [ResidualUnit(filters, initializer, activation) for _ in range(stem_layers2)]
            # transfer
            self._transfer_branch = ActorBranch(filters, initializer, activation, branch_layers2)
            self._transfer = keras.layers.Dense(2, activation="softmax", kernel_initializer=initializer_random)
            # transfer direction
            self._transfer_direction_branch = ActorBranch(filters, initializer, activation, branch_layers2)
            self._transfer_direction = keras.layers.Dense(4, activation="softmax",
                                                          kernel_initializer=initializer_random)
            # resource to transfer
            self._transfer_resource_branch = ActorBranch(filters, initializer, activation, branch_layers2)
            self._transfer_resource = keras.layers.Dense(3, activation="softmax",
                                                         kernel_initializer=initializer_random)

        def call(self, inputs, training=False, mask=None):
            features = inputs
            x = features
            center = features[:, :, :, :1]

            x = self._root(x)
            x = self._root_norm(x, training=training)
            x = self._root_activation(x)

            x1 = x
            for layer in self._stem1:
                x1 = layer(x1, training=training)
            z1 = (x1, center)

            wn = self._north_branch(z1, training=training)
            north_switch = self._north(wn)
            we = self._east_branch(z1, training=training)
            east_switch = self._east(we)
            ws = self._south_branch(z1, training=training)
            south_switch = self._south(ws)
            ww = self._west_branch(z1, training=training)
            west_switch = self._west(ww)

            wb = self._build_branch(z1, training=training)
            build_switch = self._build(wb)

            wi = self._idle_branch(z1, training=training)
            idle_switch = self._idle(wi)

            x2 = x
            for layer in self._stem2:
                x2 = layer(x2, training=training)
            z2 = (x2, center)

            wt = self._transfer_branch(z2, training=training)
            transfer_switch = self._transfer(wt)

            w_td = self._transfer_direction_branch(z2, training=training)
            transfer_direction_probs = self._transfer_direction(w_td)

            w_tr = self._transfer_resource_branch(z2, training=training)
            transfer_resource_probs = self._transfer_resource(w_tr)

            return north_switch, east_switch, south_switch, west_switch, build_switch, idle_switch, \
                   transfer_switch, transfer_direction_probs, transfer_resource_probs

        def get_config(self):
            pass

    model = ResidualModel()
    return model


def actor_critic_efficient_six_actions(actions_shape):
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
        def __init__(self, actions_n, **kwargs):
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
            self._workers_probs1 = keras.layers.Dense(actions_n, activation="softmax",
                                                      kernel_initializer=initializer_random)
            self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                                activation=keras.activations.tanh)

        def call(self, inputs, training=False, mask=None):
            outputs = self._stem(inputs, training)
            for idx, block in enumerate(self._blocks):
                survival_prob = self._mconfig.survival_prob
                if survival_prob:
                    drop_rate = 1.0 - survival_prob
                    survival_prob = 1.0 - drop_rate * float(idx) / len(self._blocks)
                # survival_prob = 1.0
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
            w = self._workers_probs1(w)
            probs = w

            baseline = self._baseline(tf.concat([y, z], axis=1))

            return probs, baseline

        def get_config(self):
            pass

    model = EfficientModel(actions_shape)
    return model


def actor_critic_efficient_shrub(actions_shape):
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
            # action type
            self._workers_probs1_0 = keras.layers.Dense(actions_number[0][0], activation="softmax",
                                                        kernel_initializer=initializer_random)
            # movement direction
            self._workers_probs1_1 = keras.layers.Dense(actions_number[1][0], activation="softmax",
                                                        kernel_initializer=initializer_random)
            # transfer direction
            self._workers_probs1_2 = keras.layers.Dense(actions_number[1][0], activation="softmax",
                                                        kernel_initializer=initializer_random)
            # resource to transfer
            self._workers_probs1_3 = keras.layers.Dense(actions_number[2][0], activation="softmax",
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
            probs0 = self._workers_probs1_0(w)
            probs1 = self._workers_probs1_1(w)
            probs2 = self._workers_probs1_2(w)
            probs3 = self._workers_probs1_3(w)

            baseline = self._baseline(tf.concat([y, z], axis=1))

            return probs0, probs1, probs2, probs3, baseline

        def get_config(self):
            pass

    model = EfficientModel(actions_shape)
    # model = eff_model.get_model("efficientnetv2-s", weights=None)
    return model
