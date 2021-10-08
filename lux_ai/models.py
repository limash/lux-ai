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
    import tensorflow as tf
    import tensorflow.keras as keras

    def swish(x):
        return x * tf.nn.sigmoid(x)

    class SEBlock(tf.keras.layers.Layer):
        def __init__(self, input_channels, ratio=0.25):
            super(SEBlock, self).__init__()
            self.num_reduced_filters = max(1, int(input_channels * ratio))
            self.pool = tf.keras.layers.GlobalAveragePooling2D()
            self.reduce_conv = tf.keras.layers.Conv2D(filters=self.num_reduced_filters,
                                                      kernel_size=(1, 1),
                                                      strides=1,
                                                      padding="same")
            self.expand_conv = tf.keras.layers.Conv2D(filters=input_channels,
                                                      kernel_size=(1, 1),
                                                      strides=1,
                                                      padding="same")

        def call(self, inputs, **kwargs):
            branch = self.pool(inputs)
            branch = tf.expand_dims(input=branch, axis=1)
            branch = tf.expand_dims(input=branch, axis=1)
            branch = self.reduce_conv(branch)
            branch = swish(branch)
            branch = self.expand_conv(branch)
            branch = tf.nn.sigmoid(branch)
            output = inputs * branch
            return output

    class MBConv(tf.keras.layers.Layer):
        def __init__(self, in_channels, out_channels, expansion_factor, stride, k, drop_connect_rate):
            super(MBConv, self).__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.stride = stride
            self.drop_connect_rate = drop_connect_rate
            self.conv1 = tf.keras.layers.Conv2D(filters=in_channels * expansion_factor,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same")
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.dwconv = tf.keras.layers.DepthwiseConv2D(kernel_size=(k, k),
                                                          strides=stride,
                                                          padding="same")
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.se = SEBlock(input_channels=in_channels * expansion_factor)
            self.conv2 = tf.keras.layers.Conv2D(filters=out_channels,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same")
            self.bn3 = tf.keras.layers.BatchNormalization()
            self.dropout = tf.keras.layers.Dropout(rate=drop_connect_rate)

        def call(self, inputs, training=None, **kwargs):
            x = self.conv1(inputs)
            x = self.bn1(x, training=training)
            x = swish(x)
            x = self.dwconv(x)
            x = self.bn2(x, training=training)
            x = self.se(x)
            x = swish(x)
            x = self.conv2(x)
            x = self.bn3(x, training=training)
            if self.stride == 1 and self.in_channels == self.out_channels:
                if self.drop_connect_rate:
                    x = self.dropout(x, training=training)
                x = tf.keras.layers.add([x, inputs])
            return x

    class EfficientModel(keras.Model):
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

            self._efficient_block = [MBConv(in_channels=filters,
                                            out_channels=filters,
                                            expansion_factor=1,
                                            stride=1,
                                            k=3,
                                            drop_connect_rate=0) for _ in range(layers)]

            self._depthwise = keras.layers.DepthwiseConv2D(13)
            self._flatten = keras.layers.Flatten()

            self._workers_probs0 = keras.layers.Dense(128, activation=activation, kernel_initializer=initializer)
            self._workers_probs1 = keras.layers.Dense(actions_number, activation="softmax",
                                                      kernel_initializer=initializer_random)

            self._baseline = keras.layers.Dense(1, kernel_initializer=initializer_random,
                                                activation=keras.activations.tanh)

        def call(self, inputs, training=False, mask=None):
            features = inputs
            x = features

            x = self._conv(x)
            x = self._norm(x, training=training)
            x = self._activation(x)

            for layer in self._efficient_block:
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

            w = self._workers_probs0(z)
            w = self._workers_probs1(w)
            probs = w

            baseline = self._baseline(tf.concat([y, z], axis=1))

            return probs, baseline

        def get_config(self):
            pass

    model = EfficientModel(actions_shape)
    return model
