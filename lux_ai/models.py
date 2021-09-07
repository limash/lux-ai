# move all imports inside functions to use ray.remote multitasking


def get_actor_critic(features_shape, actions_shape):
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
