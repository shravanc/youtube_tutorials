import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

url = 'https://raw.githubusercontent.com/shravanc/datasets/master/energy_efficiency.csv'
df = pd.read_csv(url)
print(df.head())

TARGETS = ['Y1', 'Y2']

# Split data into Train and Valid
train_df = df.sample(frac=0.85, random_state=2)
valid_df = df.drop(train_df.index)

train_y1 = train_df.pop(TARGETS[0]).values
train_y2 = train_df.pop(TARGETS[1]).values

valid_y1 = valid_df.pop(TARGETS[0]).values
valid_y2 = valid_df.pop(TARGETS[1]).values

# Scaling Inputs
stats = train_df.describe().transpose()
train_df = (train_df - stats['mean']) / stats['std']
valid_df = (valid_df - stats['mean']) / stats['std']

# scale output as well
norm_train_y1 = (train_y1 - train_y1.mean()) / train_y1.std()
norm_train_y2 = (train_y2 - train_y2.mean()) / train_y2.std()

norm_valid_y1 = (valid_y1 - valid_y1.mean()) / valid_y1.std()
norm_valid_y2 = (valid_y2 - valid_y2.mean()) / valid_y2.std()

# Converting to numpy
train_data = train_df.to_numpy()
valid_data = valid_df.to_numpy()


# Convert to tf datasets
def prepare_dataset(data, labels, batch, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


# prepare train and valid dataset
batch_size = 32
buffer = 50
train_dataset = prepare_dataset(train_data, train_y1, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_y1, batch_size, buffer)

norm_train_dataset = prepare_dataset(train_data, norm_train_y1, batch_size, buffer)
norm_valid_dataset = prepare_dataset(valid_data, norm_valid_y1, batch_size, buffer)

# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.mse,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['accuracy', 'mae']
)

# print("*********************Training without Normalisation******************")
# history = model.fit(
#     train_dataset,
#     epochs=10,
#     validation_data=valid_dataset
# )

print("*********************Training with Normalisation*********************")
norm_history = model.fit(
    norm_train_dataset,
    epochs=10,
    validation_data=norm_valid_dataset
)

# plotting graphs
plots = ['accuracy', 'mae', 'loss']
for plot in plots:
    metric = norm_history.history[plot]
    val_metric = norm_history.history[f"val_{plot}"]

    epochs = range(len(metric))

    plt.figure(figsize=(15, 10))
    plt.plot(epochs, metric, label=f"Training {plot}")
    plt.plot(epochs, val_metric, label=f"Validation {plot}")
    plt.legend()
    # plt.show()

forecast = []
for index, row in valid_df.iterrows():
    data = row.values[np.newaxis]
    prediction = model.predict(data)
    forecast.append(prediction[0][0])


valid = norm_valid_y1[-50:]
forecast = forecast[-50:]
time = range(len(valid))

plt.figure(figsize=(15, 10))
plt.plot(time, valid, label="Original Plot")
plt.plot(time, forecast, label="Prediction Plot")
plt.legend()
# plt.show()


model_mae = tf.keras.losses.mean_squared_error(valid, forecast)
print("Model MAE---->", model_mae)