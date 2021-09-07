# move all imports inside functions to use ray.remote multitasking


def get_actor_critic_test(features_shape, actions_shape):
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers

    conv_initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
    initializer_random = keras.initializers.random_uniform(minval=-0.01, maxval=0.01)

    input_A = layers.Input(shape=features_shape, name="features_input")
    input_B = layers.Input(shape=actions_shape, name="actions_mask_input")

    x = layers.Conv2D(filters=32, kernel_size=1, activation="relu", kernel_initializer=conv_initializer)(input_A)

    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)  # 30
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)  # 20
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)  # 10
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu", kernel_initializer=conv_initializer)(x)  # 2

    x = layers.Flatten()(x)

    city_tiles_probs = layers.Dense(4, activation="softmax", kernel_initializer=initializer_random)(x)
    workers_probs = layers.Dense(19, activation="softmax", kernel_initializer=initializer_random)(x)
    carts_probs = layers.Dense(17, activation="softmax", kernel_initializer=initializer_random)(x)

    all_probs = layers.Concatenate()([city_tiles_probs, workers_probs, carts_probs])
    probs = layers.Multiply(name="probs_output")([all_probs, input_B])
    baseline = layers.Dense(1, activation="tanh", name="value_output", kernel_initializer=initializer_random)(x)

    model = keras.Model(inputs=[input_A, input_B], outputs=[probs, baseline])
    return model


def get_actor_critic():
    import tensorflow as tf
    import tensorflow.keras as keras

    def circular_padding(x):
        x = tf.concat([x[:, -1:, :, :], x, x[:, :1, :, :]], 1)
        x = tf.concat([x[:, :, -1:, :], x, x[:, :, :1, :]], 2)
        return x

    class ResidualUnit(keras.layers.Layer):
        def __init__(self, filters, initializer, activation, **kwargs):
            super().__init__(**kwargs)

            self._filters = filters
            self._activation = activation
            self._conv = keras.layers.Conv2D(filters, 3, kernel_initializer=initializer, use_bias=False)
            self._norm = keras.layers.BatchNormalization()

        def call(self, inputs, training=False, **kwargs):
            x = inputs
            x = circular_padding(x)
            x = self._conv(x)
            x = self._norm(x, training=training)
            return self._activation(inputs + x)

        def compute_output_shape(self, batch_input_shape):
            batch, x, y, _ = batch_input_shape
            return [batch, x, y, self._filters]

    class SmallResidualModel(keras.Model):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

            filters = 64
            layers = 12

            initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
            activation = keras.activations.relu

            self._conv = keras.layers.Conv2D(filters, 3, kernel_initializer=initializer)
            self._norm = keras.layers.BatchNormalization()
            self._activation = keras.layers.ReLU()
            self._residual_block = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]

            self._city_tiles_probs0 = keras.layers.Dense(64, activation=activation, kernel_initializer=initializer)
            self._city_tiles_probs1 = keras.layers.Dense(4, activation="softmax", kernel_initializer=initializer_random)
            self._workers_probs0 = keras.layers.Dense(64, activation=activation, kernel_initializer=initializer)
            self._workers_probs1 = keras.layers.Dense(19, activation="softmax", kernel_initializer=initializer_random)
            self._carts_probs0 = keras.layers.Dense(64, activation=activation, kernel_initializer=initializer)
            self._carts_probs1 = keras.layers.Dense(17, activation="softmax", kernel_initializer=initializer_random)

            self._probs = keras.layers.Multiply(name="probs_output")
            self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                                activation=keras.activations.tanh,
                                                name="value_output")

        def call(self, inputs, training=False, mask=None):
            features, actions_mask = inputs

            x = features

            x = circular_padding(x)
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
            probs = self._probs([probs, actions_mask])

            baseline = self._baseline(tf.concat([y, z], axis=1))

            return probs, baseline

        def get_config(self):
            pass

    model = SmallResidualModel()
    return model
