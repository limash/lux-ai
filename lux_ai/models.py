# move all imports inside functions to use ray.remote multitasking


def actor_critic_separate():
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
            layers = 6

            initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
            activation = keras.activations.relu

            self._conv = keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, use_bias=False)
            self._norm = keras.layers.BatchNormalization()
            self._activation = keras.layers.ReLU()
            self._residual_block = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]
            self._residual_block_tile = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]
            self._residual_block_work = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]
            self._residual_block_cart = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]

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
            batch_size = actions_mask.shape[0]

            is_tile = actions_mask[:, 0]
            is_work = actions_mask[:, 4]
            is_cart = actions_mask[:, -1]

            x = features
            tile_features = tf.gather(features, indices=tf.where(is_tile))
            work_features = tf.gather(features, indices=tf.where(is_work))
            cart_features = tf.gather(features, indices=tf.where(is_cart))

            x = self._conv(x)
            x = self._norm(x, training=training)
            x = self._activation(x)

            for layer in self._residual_block:
                x = layer(x, training=training)

            # tiles
            tile = tf.gather(x, indices=tf.where(is_tile))
            if tile.shape[0] != 0:
                tile = tile[:, 0, :]
                for layer in self._residual_block_tile:
                    tile = layer(tile, training=training)

                z_tile = tile * tile_features[:, 0, :, :, :1]
                shape_z_tile = tf.shape(z_tile)
                z_tile = tf.reshape(z_tile, (shape_z_tile[0], -1, shape_z_tile[-1]))
                z_tile = tf.reduce_sum(z_tile, axis=1)
                t = self._city_tiles_probs0(z_tile)
                t = self._city_tiles_probs1(t)

                t_shape = [batch_size, 4]
                indices = tf.where(is_tile)
                data = t
                tt = tf.scatter_nd(indices=indices, updates=data, shape=t_shape)
            else:
                tt = actions_mask[:, :4]

            # workers
            work = tf.gather(x, indices=tf.where(is_work))
            if work.shape[0] != 0:
                work = work[:, 0, :]
                for layer in self._residual_block_work:
                    work = layer(work, training=training)

                z_work = work * work_features[:, 0, :, :, :1]
                shape_z_work = tf.shape(z_work)
                z_work = tf.reshape(z_work, (shape_z_work[0], -1, shape_z_work[-1]))
                z_work = tf.reduce_sum(z_work, axis=1)
                w = self._workers_probs0(z_work)
                w = self._workers_probs1(w)

                w_shape = [batch_size, 19]
                indices = tf.where(is_work)
                data = w
                ww = tf.scatter_nd(indices=indices, updates=data, shape=w_shape)
            else:
                ww = actions_mask[:, 4:23]

            # carts
            cart = tf.gather(x, indices=tf.where(is_cart))
            if cart.shape[0] != 0:
                cart = cart[:, 0, :]
                for layer in self._residual_block_cart:
                    cart = layer(cart, training=training)

                z_cart = cart * cart_features[:, 0, :, :, :1]
                shape_z_cart = tf.shape(z_cart)
                z_cart = tf.reshape(z_cart, (shape_z_cart[0], -1, shape_z_cart[-1]))
                z_cart = tf.reduce_sum(z_cart, axis=1)
                c = self._carts_probs0(z_cart)
                c = self._carts_probs1(c)

                c_shape = [batch_size, 17]
                indices = tf.where(is_cart)
                data = c
                cc = tf.scatter_nd(indices=indices, updates=data, shape=c_shape)
            else:
                cc = actions_mask[:, 23:]

            probs = tf.concat([tt, ww, cc], axis=1)

            # value
            shape_x = tf.shape(x)
            y = tf.reshape(x, (shape_x[0], -1, shape_x[-1]))
            y = tf.reduce_mean(y, axis=1)

            z = (x * features[:, :, :, :1])
            shape_z = tf.shape(z)
            z = tf.reshape(z, (shape_z[0], -1, shape_z[-1]))
            z = tf.reduce_sum(z, axis=1)

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

            self._residual_block3 = [ResidualUnit(filters, initializer, activation) for _ in range(4)]

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

            for layer in self._residual_block3:
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


def actor_critic_squeeze(actions_shape):
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

            filters = 64
            layers = 12

            initializer = keras.initializers.VarianceScaling(scale=2.0, mode='fan_in', distribution='truncated_normal')
            initializer_random = keras.initializers.random_uniform(minval=-0.03, maxval=0.03)
            activation = keras.activations.relu

            self._conv = keras.layers.Conv2D(filters, 3, padding="same", kernel_initializer=initializer, use_bias=False)
            self._norm = keras.layers.BatchNormalization()
            self._activation = keras.layers.ReLU()
            self._residual_block = [ResidualUnit(filters, initializer, activation) for _ in range(layers)]

            self._depthwise = keras.layers.DepthwiseConv2D(32)
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
            features = tf.concat([inputs[:, :, :, :1],
                                  inputs[:, :, :, 4:10],
                                  inputs[:, :, :, 15:42],
                                  inputs[:, :, :, 43:44],
                                  inputs[:, :, :, 45:],
                                  ], axis=-1)
            batch_size = features.shape[0]

            features_padded = tf.pad(features, tf.constant([[0, 0], [6, 6], [6, 6], [0, 0]]), mode="CONSTANT")
            units_layers = features_padded[:, :, :, :1]
            units_coords = tf.cast(tf.where(units_layers), dtype=tf.int32)
            min_x = units_coords[:, 1] - 6
            max_x = units_coords[:, 1] + 6
            min_y = units_coords[:, 2] - 6
            max_y = units_coords[:, 2] + 6

            features_padded_glob = tf.pad(features,
                                          tf.constant([[0, 0], [32, 32], [32, 32], [0, 0]]),
                                          mode="CONSTANT")
            units_layers_glob = features_padded_glob[:, :, :, :1]
            units_coords_glob = tf.cast(tf.where(units_layers_glob), dtype=tf.int32)
            min_x_glob = units_coords_glob[:, 1] - 32
            max_x_glob = units_coords_glob[:, 1] + 32
            min_y_glob = units_coords_glob[:, 2] - 32
            max_y_glob = units_coords_glob[:, 2] + 32

            features_size = features.shape[-1]
            t_shape = tf.constant([batch_size, 13, 13, -1])
            features_v = features.numpy()
            features_padded_v = features_padded.numpy()
            features_padded_glob_v = features_padded_glob.numpy()

            ta = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=False)
            ta_glob = tf.TensorArray(dtype=tf.float32, size=batch_size, dynamic_size=False)
            for i in tf.range(batch_size):
                piece = features_padded[i, min_x[i]: max_x[i] + 1, min_y[i]: max_y[i] + 1, :]
                piece_v = piece.numpy()
                ta = ta.write(i, piece)
                piece_glob = features_padded_glob[i:i + 1,
                                                  min_x_glob[i]: max_x_glob[i] + 1,
                                                  min_y_glob[i]: max_y_glob[i] + 1,
                                                  37:46]
                # 17, 4; 5, 5
                pooled_features = tf.nn.avg_pool(piece_glob, 5, 5, padding="VALID")
                piece_glob_v = piece_glob.numpy()
                pooled_features_v = pooled_features.numpy()
                ta_glob = ta_glob.write(i, pooled_features[0, :, :, :])

            features_prepared_1 = ta.stack()
            features_prepared_2 = ta_glob.stack()
            features_prepared = tf.concat([features_prepared_1, features_prepared_2], axis=-1)

            x = features_prepared

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
            probs = w

            baseline = self._baseline(tf.concat([y, z], axis=1))

            return probs, baseline

        def get_config(self):
            pass

    model = ResidualModel(actions_shape)
    return model


def actor_critic_base(actions_shape):
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

            filters = 128
            layers = 12

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
            # features = tf.concat([inputs[:, :, :, :1],
            #                       inputs[:, :, :, 4:10],
            #                       inputs[:, :, :, 15:42],
            #                       inputs[:, :, :, 43:44],
            #                       inputs[:, :, :, 45:],
            #                       ], axis=-1)

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
