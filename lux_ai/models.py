# move all imports inside functions to use ray.remote multitasking

def norm_probs(probs_unnorm):
    import tensorflow as tf

    logits = tf.math.log(probs_unnorm)
    probs = tf.nn.softmax(logits)
    return probs


def get_actor_critic(features_shape, actions_shape):
    import tensorflow.keras as keras
    import tensorflow.keras.layers as layers

    input_A = layers.Input(shape=features_shape, name="features_input")
    input_B = layers.Input(shape=actions_shape, name="actions_mask_input")

    x = layers.Conv2D(filters=32, kernel_size=1, activation="relu")(input_A)

    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)  # 30
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)  # 20
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)  # 10
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
    x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)  # 2

    x = layers.Flatten()(x)

    all_probs = layers.Dense(actions_shape, activation="softmax")(x)
    spec_probs_unnorm = layers.Multiply()([all_probs, input_B])
    probs = layers.Lambda(norm_probs, name="probs_output")(spec_probs_unnorm)

    baseline = layers.Dense(1, activation="tanh", name="value_output")(x)

    model = keras.Model(inputs=[input_A, input_B], outputs=[probs, baseline])
    return model
