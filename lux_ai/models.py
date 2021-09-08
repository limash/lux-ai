# move all imports inside functions to use ray.remote multitasking


def actor_critic_1():
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


def actor_critic_2(features_shape, actions_shape):
    # import tensorflow as tf
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers

    initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
    initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)

    input_A = layers.Input(shape=features_shape, name="features_input")
    input_B = layers.Input(shape=actions_shape, name="actions_mask_input")

    # projecting
    # x = layers.Conv2D(filters=32, kernel_size=1, activation="relu", kernel_initializer=initializer)(input_A)
    x = input_A

    # downsampling
    # x = layers.Conv2D(filters=64, kernel_size=4, activation="relu", strides=2, kernel_initializer=initializer)(x)

    # z = x
    z = layers.Conv2D(filters=64, kernel_size=3, padding="same", kernel_initializer=initializer)(x)
    z = keras.layers.BatchNormalization()(z)
    z = layers.Activation("relu")(z)

    # res block 1
    def res_block(a):
        b = layers.Conv2D(filters=64, kernel_size=3, padding="same", kernel_initializer=initializer, use_bias=False)(a)
        b = keras.layers.BatchNormalization()(b)
        b = layers.Add()([b, a])
        return layers.Activation("relu")(b)

    for _ in range(12):
        z = res_block(z)

    # skip connection from input
    # z = layers.Add()([z, x])
    # z = keras.layers.BatchNormalization()(z)
    # z = layers.Activation("relu")(z)

    z = layers.Multiply()([z, x[:, :, :, :1]])
    z = layers.MaxPooling2D(32)(z)
    z = layers.Flatten()(z)

    # city tiles
    z_t = keras.layers.Dense(64, activation="relu", kernel_initializer=initializer)(z)
    # z_t = keras.layers.Dense(256, activation="relu", kernel_initializer=initializer)(z_t)
    city_tiles_probs = keras.layers.Dense(4, activation="softmax", kernel_initializer=initializer_random)(z_t)

    # workers
    z_w = keras.layers.Dense(64, activation="relu", kernel_initializer=initializer)(z)
    # z_w = keras.layers.Dense(256, activation="relu", kernel_initializer=initializer)(z_w)
    workers_probs = keras.layers.Dense(19, activation="softmax", kernel_initializer=initializer_random)(z_w)

    # carts
    z_c = keras.layers.Dense(64, activation="relu", kernel_initializer=initializer)(z)
    # z_c = keras.layers.Dense(256, activation="relu", kernel_initializer=initializer)(z_c)
    carts_probs = keras.layers.Dense(17, activation="softmax", kernel_initializer=initializer_random)(z_c)

    all_probs = layers.Concatenate()([city_tiles_probs, workers_probs, carts_probs])
    probs = layers.Multiply(name="output_1")([all_probs, input_B])
    baseline = layers.Dense(1, activation="tanh", name="output_2", kernel_initializer=initializer_random)(z)

    model = keras.Model(inputs=[input_A, input_B], outputs=[probs, baseline])
    return model
