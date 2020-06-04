import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


url = 'https://raw.githubusercontent.com/shravanc/datasets/master/graduate_admission.csv'
df = pd.read_csv(url)
df = df.drop('index', axis=1)


train_df = df.sample(frac=.8, random_state=0)
valid_df = df.drop(train_df.index)

TARGET = 'admit'

train_labels = train_df.pop(TARGET).values
valid_labels = valid_df.pop(TARGET).values

train_stats = train_df.describe().transpose()

norm_train = (train_df-train_stats['mean'])/train_stats['std']
norm_valid = (valid_df-train_stats['mean'])/train_stats['std']


train_data = norm_train.to_numpy()
valid_data = norm_valid.to_numpy()


def prepare_dataset(data, label, batch, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


batch_size = 32
buffer = 100

train_dataset = prepare_dataset(train_data, train_labels, batch_size, shuffle_buffer=buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, shuffle_buffer=buffer)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


model.compile(
    loss='mse',
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['mae']
)

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=valid_dataset
)


mae = history.history['mae']
val_mae = history.history['val_mae']

epochs = range(len(mae))

plt.figure(figsize=(15, 10))
plt.plot(epochs, mae, label='Training MAE')
plt.plot(epochs, val_mae, label='Validation MAE')
plt.legend(loc='upper left')
plt.show()