import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import IPython
import kerastuner as kt
from tensorflow import keras

# Load Data to DataFrame
url = 'https://raw.githubusercontent.com/shravanc/datasets/master/graduate_admission.csv'
df = pd.read_csv(url)
df = df.drop('index', axis=1)
print(df.head())

# Categorical Columns
categorical_column = ['research']
df = pd.get_dummies(df, columns=categorical_column)
print(df.head())

# Separate Data to Train and Validation set
train_df = df.sample(frac=0.8)
valid_df = df.drop(train_df.index)

# Labels columns
TARGET = 'admit'
train_labels = train_df.pop(TARGET).values
valid_labels = valid_df.pop(TARGET).values

# Extract first column to check prediction
prediction_data = train_df.iloc[0].values
result = train_labels[0]

# Scaling data
stats = train_df.describe().transpose()
norm_train_df = (train_df - stats['mean']) / stats['std']
norm_valid_df = (valid_df - stats['mean']) / stats['std']
print(norm_train_df.head())

# Converting to numpy array
train_data = norm_train_df.to_numpy()
valid_data = norm_valid_df.to_numpy()


# Prepare tf.data.Dataset for training and validating
def prepare_dataset(data, labels, batch, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


# Training and Validation Dataset
batch_size = 32
buffer = 100
train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)


def model_builder(hp):
    model = keras.Sequential()
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(1))

    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.mse,
                  metrics=['accuracy'])

    return model


tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')


class ClearTrainingOutput(tf.keras.callbacks.Callback):
    def on_train_end(*args, **kwargs):
        IPython.display.clear_output(wait=True)


tuner.search(train_dataset, epochs=10, validation_data=valid_dataset, callbacks=[ClearTrainingOutput()])

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")
