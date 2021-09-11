# move all imports inside functions to use ray.remote multitasking


def actor_critic():
    import tensorflow as tf
    import tensorflow.keras as keras

    def height_padding(x, height):
        x = tf.concat([x[:, -height:, :, :], x, x[:, :height, :, :]], 1)
        return x

    def width_padding(x, width):
        x = tf.concat([x[:, :, -width:, :], x, x[:, :, :width, :]], 2)
        return x

    class CrossUnit(keras.layers.Layer):
        def __init__(self, filters, initializer, size, **kwargs):
            super().__init__(**kwargs)

            self._height, self._width = size
            self._filters = filters
            self._conv_col = keras.layers.Conv2D(filters/4, (self._height, 1),
                                                 kernel_initializer=initializer, use_bias=False)
            self._conv_row = keras.layers.Conv2D(filters/4, (1, self._width),
                                                 kernel_initializer=initializer, use_bias=False)
            self._conv = keras.layers.Conv2D(filters/2, 3, kernel_initializer=initializer, use_bias=False)
            self._norm = keras.layers.BatchNormalization()

        def call(self, inputs, training=False, **kwargs):
            x = inputs
            y = height_padding(x, self._height//2)
            y = self._conv_col(y)
            z = width_padding(x, self._width//2)
            z = self._conv_row(z)
            o = circular_padding(x)
            o = self._conv(o)
            # x = y + z
            x = tf.concat([y, z, o], axis=-1)
            outputs = self._norm(x, training=training)
            return outputs

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
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            filters = 128
            layers = 12

            initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
            activation = keras.activations.relu

            self._conv = keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, use_bias=False)
            self._norm = keras.layers.BatchNormalization()
            self._activation = keras.layers.ReLU()
            self._residual_block = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]

            self._city_tiles_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
            self._city_tiles_probs1 = keras.layers.Dense(4, activation="softmax", kernel_initializer=initializer_random)
            self._workers_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
            self._workers_probs1 = keras.layers.Dense(19, activation="softmax", kernel_initializer=initializer_random)
            self._carts_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
            self._carts_probs1 = keras.layers.Dense(17, activation="softmax", kernel_initializer=initializer_random)

            self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                                activation=keras.activations.tanh)

        def call(self, inputs, training=False, mask=None):
            features, actions_mask = inputs
            features_v = features.numpy()
            actions_mask_v = actions_mask.numpy()

            x = features

            x = self._conv(x)
            x = self._norm(x, training=training)
            x = self._activation(x)

            for layer in self._residual_block:
                x = layer(x, training=training)

            shape_x = tf.shape(x)
            y = tf.reshape(x, (shape_x[0], -1, shape_x[-1]))
            y = tf.reduce_mean(y, axis=1)

            z = (x * features[:, :, :, :1])
            shape_z = tf.shape(z)
            z = tf.reshape(z, (shape_z[0], -1, shape_z[-1]))
            z = tf.reduce_sum(z, axis=1)

            t = self._city_tiles_probs0(z)
            t = self._city_tiles_probs1(t)
            w = self._workers_probs0(z)
            w = self._workers_probs1(w)
            c = self._carts_probs0(z)
            c = self._carts_probs1(c)
            probs = tf.concat([t, w, c], axis=1)
            probs = probs * actions_mask

            baseline = self._baseline(tf.concat([y, z], axis=1))

            return probs, baseline

        def get_config(self):
            pass

    model = ResidualModel()
    return model


def actor_critic_with_skip_connections():
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
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            filters = 128

            initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
            activation = keras.activations.relu

            self._conv = keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, use_bias=False)
            self._norm = keras.layers.BatchNormalization()
            self._activation = keras.layers.ReLU()

            self._residual_block1 = [ResidualUnit(filters, initializer, activation) for _ in range(3)]
            self._conv1 = keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer,
                                              use_bias=False)
            self._norm1 = keras.layers.BatchNormalization()
            self._activation1 = keras.layers.ReLU()

            self._residual_block2 = [ResidualUnit(filters, initializer, activation) for _ in range(3)]
            self._conv2 = keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer,
                                              use_bias=False)
            self._norm2 = keras.layers.BatchNormalization()
            self._activation2 = keras.layers.ReLU()

            self._residual_block3 = [ResidualUnit(filters, initializer, activation) for _ in range(3)]
            self._conv3 = keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer,
                                              use_bias=False)
            self._norm3 = keras.layers.BatchNormalization()
            self._activation3 = keras.layers.ReLU()

            self._city_tiles_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
            self._city_tiles_probs1 = keras.layers.Dense(4, activation="softmax", kernel_initializer=initializer_random)
            self._workers_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
            self._workers_probs1 = keras.layers.Dense(19, activation="softmax", kernel_initializer=initializer_random)
            self._carts_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
            self._carts_probs1 = keras.layers.Dense(17, activation="softmax", kernel_initializer=initializer_random)

            self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                                activation=keras.activations.tanh)

        def call(self, inputs, training=False, mask=None):
            features, actions_mask = inputs

            x = features

            x = self._conv(x)
            x = self._norm(x, training=training)
            x = self._activation(x)

            s = x
            for layer in self._residual_block1:
                x = layer(x, training=training)
            x = self._conv1(x)
            x = self._norm1(x, training=training)
            x = self._activation1(s + x)

            s = x
            for layer in self._residual_block2:
                x = layer(x, training=training)
            x = self._conv2(x)
            x = self._norm2(x, training=training)
            x = self._activation2(s + x)

            s = x
            for layer in self._residual_block3:
                x = layer(x, training=training)
            x = self._conv3(x)
            x = self._norm3(x, training=training)
            x = self._activation3(s + x)

            shape_x = tf.shape(x)
            y = tf.reshape(x, (shape_x[0], -1, shape_x[-1]))
            y = tf.reduce_mean(y, axis=1)

            z = (x * features[:, :, :, :1])
            shape_z = tf.shape(z)
            z = tf.reshape(z, (shape_z[0], -1, shape_z[-1]))
            z = tf.reduce_sum(z, axis=1)

            t = self._city_tiles_probs0(z)
            t = self._city_tiles_probs1(t)
            w = self._workers_probs0(z)
            w = self._workers_probs1(w)
            c = self._carts_probs0(z)
            c = self._carts_probs1(c)
            probs = tf.concat([t, w, c], axis=1)
            probs = probs * actions_mask

            baseline = self._baseline(tf.concat([y, z], axis=1))

            return probs, baseline

        def get_config(self):
            pass

    model = ResidualModel()
    return model


def actor_critic_custom():
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
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            filters = 128
            layers = 12

            initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
            activation = keras.activations.relu

            self._conv = keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, use_bias=False)
            self._norm = keras.layers.BatchNormalization()
            self._activation = keras.layers.ReLU()
            self._residual_block = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]

            self._city_tiles_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
            self._city_tiles_probs1 = keras.layers.Dense(4, activation="softmax", kernel_initializer=initializer_random)
            self._workers_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
            self._workers_probs1 = keras.layers.Dense(19, activation="softmax", kernel_initializer=initializer_random)
            self._carts_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
            self._carts_probs1 = keras.layers.Dense(17, activation="softmax", kernel_initializer=initializer_random)

            self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                                activation=keras.activations.tanh)

        def call(self, inputs, training=False, mask=None):
            features, actions_mask = inputs

            x = features

            x = self._conv(x)
            x = self._norm(x, training=training)
            x = self._activation(x)

            for layer in self._residual_block:
                x = layer(x, training=training)

            shape_x = tf.shape(x)
            y = tf.reshape(x, (shape_x[0], -1, shape_x[-1]))
            y = tf.reduce_mean(y, axis=1)

            z = (x * features[:, :, :, :1])
            shape_z = tf.shape(z)
            z = tf.reshape(z, (shape_z[0], -1, shape_z[-1]))
            z = tf.reduce_sum(z, axis=1)

            t = self._city_tiles_probs0(z)
            t = self._city_tiles_probs1(t)
            w = self._workers_probs0(z)
            w = self._workers_probs1(w)
            c = self._carts_probs0(z)
            c = self._carts_probs1(c)
            probs = tf.concat([t, w, c], axis=1)
            probs = probs * actions_mask

            baseline = self._baseline(tf.concat([y, z], axis=1))

            return probs, baseline

        def get_config(self):
            pass

    model = ResidualModel()
    return model
