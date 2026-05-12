import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, Dense, Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Add, concatenate
from tensorflow.keras.layers import Activation

def baseline_model(inputs, angles, charges, input_dim, dropout_level=0.10):
    inputs_flat = Flatten()(inputs)
    concat_inputs = concatenate([inputs_flat, angles, charges])

    # Position branch
    position = Dense(8 * input_dim)(concat_inputs)
    position = LeakyReLU(negative_slope=0.01)(position)
    position = Dense(4 * input_dim)(position)
    position = LeakyReLU(negative_slope=0.01)(position)
    position = BatchNormalization()(position)
    position = Dropout(dropout_level)(position)

    position_res = Add()([position, Dense(4 * input_dim)(concat_inputs)])
    position_out = Dense(1)(position_res)

    # Uncertainty branch
    uncertainty = Dense(8 * input_dim)(concat_inputs)
    uncertainty = LeakyReLU(negative_slope=0.01)(uncertainty)
    uncertainty = Dense(4 * input_dim)(uncertainty)
    uncertainty = LeakyReLU(negative_slope=0.01)(uncertainty)
    uncertainty = BatchNormalization()(uncertainty)
    uncertainty = Dropout(dropout_level)(uncertainty)

    uncertainty_res = Add()([uncertainty, Dense(4 * input_dim)(concat_inputs)])
    uncertainty_raw = Dense(1)(uncertainty_res)
    uncertainty_tanh = Activation('tanh')(uncertainty_raw)

    # Limits uncertainty to a reasonable range (5 um to 150 um)
    unc_max = 150.0*1e-4 # 150 um in cm
    unc_min = 5.0*1e-4   # 5 um in cm
    uncertainty_out = (unc_min + (unc_max - unc_min) * (1.0 + uncertainty_tanh) / 2.0)    
    position_uncertainty = concatenate([position_out, uncertainty_out])
    model = Model(inputs=[inputs, angles, charges], outputs=[position_uncertainty])
    return model

def conv_model(inputs, angles, charges, input_dim, dropout_level=0.10):
    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same")(inputs)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Dropout(dropout_level)(x)

    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.LeakyReLU(negative_slope=0.01)(x)
    x = tf.keras.layers.BatchNormalization(axis=-1)(x)
    x = tf.keras.layers.Dropout(dropout_level)(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)  # replaces Flatten

    concat_list = [x, angles, charges]
    concat_inputs = tf.keras.layers.concatenate(concat_list)

    # Position branch
    position = tf.keras.layers.Dense(8 * input_dim)(concat_inputs)
    position = tf.keras.layers.LeakyReLU(negative_slope=0.01)(position)
    position = tf.keras.layers.Dense(4 * input_dim)(position)
    position = tf.keras.layers.LeakyReLU(negative_slope=0.01)(position)
    position = tf.keras.layers.BatchNormalization()(position)
    position = tf.keras.layers.Dropout(dropout_level)(position)

    position_res = tf.keras.layers.Add()([position, tf.keras.layers.Dense(4 * input_dim)(concat_inputs)])
    position_out = tf.keras.layers.Dense(1)(position_res)

    # Uncertainty branch
    uncertainty = tf.keras.layers.Dense(8 * input_dim)(concat_inputs)
    uncertainty = tf.keras.layers.LeakyReLU(negative_slope=0.01)(uncertainty)
    uncertainty = tf.keras.layers.Dense(4 * input_dim)(uncertainty)
    uncertainty = tf.keras.layers.LeakyReLU(negative_slope=0.01)(uncertainty)
    uncertainty = tf.keras.layers.BatchNormalization()(uncertainty)
    uncertainty = tf.keras.layers.Dropout(dropout_level)(uncertainty)

    uncertainty_res = tf.keras.layers.Add()([uncertainty, tf.keras.layers.Dense(4 * input_dim)(concat_inputs)])
    uncertainty_out = tf.keras.layers.Dense(1, activation='softplus')(uncertainty_res)

    position_uncertainty = tf.keras.layers.concatenate([position_out, uncertainty_out])
    model = tf.keras.models.Model(
        inputs=[inputs, angles] if charges is None else [inputs, angles, charges],
        outputs=[position_uncertainty]
    )
    return model
