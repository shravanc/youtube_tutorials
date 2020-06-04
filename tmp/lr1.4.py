import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/shravanc/datasets/master/graduate_admission.csv'
df = pd.read_csv(url)
df = df.drop('index', axis=1)

print(df.head())

categorical_columns = ['uni_rating', 'research']
df = pd.get_dummies(df, columns=categorical_columns)


TARGET = 'admit'

pred_test_data = df.drop(TARGET, axis=1).iloc[0].values
expected_result = df.iloc[0]['admit']


train_df = df.sample(frac=0.8, random_state=0)
valid_df = df.drop(train_df.index)

train_labels = train_df.pop(TARGET)
valid_labels = valid_df.pop(TARGET)

stats = train_df.describe().transpose()

norm_train = (train_df-stats['mean']) / stats['std']
norm_valid = (valid_df-stats['mean']) / stats['std']

print(norm_train.head())

train_data = train_df.to_numpy()
valid_data = valid_df.to_numpy()


def prepare_dataset(data, labels, batch, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


batch_size = 32
buffer = 100

train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


model.compile(
    loss=tf.keras.losses.mse,
    optimizer=tf.keras.optimizers.SGD(lr=0.001, momentum=0.9),
    metrics=['mae']
)

history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=valid_dataset
)

print(pred_test_data)
print("Expected Result--->", expected_result)

mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(len(mae))

plt.figure(figsize=(15,10))
plt.plot(epochs, mae, label='MAE')
plt.plot(epochs, val_mae, label='val MAE')
plt.legend()
#plt.show()


prediction = model.predict([pred_test_data])
print(prediction)