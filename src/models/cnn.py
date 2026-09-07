import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(input_shape):

    model = models.Sequential([

        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv1D(
            filters=32,
            kernel_size=7,
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv1D(
            filters=32,
            kernel_size=7,
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.MaxPooling1D(
            pool_size=4
        ),

        # Block 2
        layers.Conv1D(
            filters=64,
            kernel_size=7,
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv1D(
            filters=64,
            kernel_size=7,
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.MaxPooling1D(
            pool_size=4
        ),

        # Block 3
        layers.Conv1D(
            filters=128,
            kernel_size=5,
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.Conv1D(
            filters=128,
            kernel_size=5,
            padding="same"
        ),
        layers.BatchNormalization(),
        layers.ReLU(),

        layers.MaxPooling1D(
            pool_size=4
        ),

        # Feature aggregation
        layers.GlobalAveragePooling1D(),

        layers.Dense(
            64,
            activation="relu"
        ),

        layers.Dropout(0.4),

        layers.Dense(
            1,
            activation="sigmoid"
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            ),

            tf.keras.metrics.AUC(
                name="auc"
            )
        ]
    )

    return model