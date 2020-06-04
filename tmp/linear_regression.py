import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


url = 'https://raw.githubusercontent.com/shravanc/datasets/master/graduate_admission.csv'
df = pd.read_csv(url)
df = df.drop('index', axis=1)
print(df.head(2))

print(df.describe())

train_df = df.sample(frac=0.8, random_state=0)
valid_df = df.drop(train_df.index)

TARGET = 'admit'

train_labels = train_df.pop(TARGET)
valid_labels = valid_df.pop(TARGET)

train_labels = train_labels.values
valid_labels = valid_labels.values

train_data = train_df.to_numpy()
valid_data = valid_df.to_numpy()


def get_dataset(data, labels, batch_size, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch_size).prefetch(1)
    return dataset


bs = 32
sb = 500
train_dataset = get_dataset(train_data, train_labels, bs, sb)
valid_dataset = get_dataset(valid_data, valid_labels, bs, sb)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1)
])

# Regression Problem
model.compile(
    loss=tf.keras.losses.mae,
    optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9),
    metrics=['mae']
)

history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=valid_dataset
)


acc = history.history['mae']
val_acc = history.history['val_mae']

epochs = range(len(acc))

plt.figure(figsize=(20, 10))
plt.plot(epochs, acc, label='Training MAE')
plt.plot(epochs, val_acc, label='Validation MAE')
plt.show()


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(20, 10))
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.show()


answer = 0.92
sample_data = "337,118,4,4.5,4.5,9.65,1"
data = [sample_data.split(',')]
print(data)

prediction = model.predict(data)
print(prediction)