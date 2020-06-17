import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


url = 'https://raw.githubusercontent.com/shravanc/datasets/master/energy_efficiency.csv'
df = pd.read_csv(url)
print(df.head())


TARGETS = ['Y1', 'Y2']

# Split the data into Train and Valid
train_df = df.sample(frac=0.85, random_state=2)
valid_df = df.drop(train_df.index)


# Separate targets
train_y1 = train_df.pop(TARGETS[0]).values
train_y2 = train_df.pop(TARGETS[1]).values

valid_y1 = valid_df.pop(TARGETS[0]).values
valid_y2 = valid_df.pop(TARGETS[1]).values


# Scaling Input features
stats = train_df.describe().transpose()
train_df = (train_df-stats['mean'])/stats['std']
valid_df = (valid_df-stats['mean'])/stats['std']


# Scaling target as well
norm_train_y1 = (train_y1-train_y1.mean())/train_y1.std()
norm_train_y2 = (train_y2-train_y2.mean())/train_y2.std()

norm_valid_y1 = (valid_y1-valid_y1.mean())/valid_y1.std()
norm_valid_y2 = (valid_y2-valid_y2.mean())/valid_y2.std()


# Converting to numpy array
train_data = train_df.to_numpy()
valid_data = valid_df.to_numpy()


# Prepare tf dataset
def prepare_dataset(data, labels, batch, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


# prepare dataset for train and valid
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
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.mse,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['accuracy', 'mae']
)

# print("*****Without Normalisation*****")
# # Training without normalisation
# history = model.fit(
#     train_dataset,
#     epochs=10,
#     validation_data=valid_dataset
# )

print("*****With Normalisation*****")
# Training with normalisation
norm_history = model.fit(
    norm_train_dataset,
    epochs=500,
    validation_data=norm_valid_dataset
)

# Without normalisation for 10 epochs mae is = 1.6694
# With normalisation for 10 epochs mae is = 0.1719

# From now on we will continue with normalised training
# Plotting the graphs
plots = ['mae', 'loss']
for plot in plots:
    metric = norm_history.history[plot]
    val_metric = norm_history.history[f"val_{plot}"]

    epochs = range(len(metric))

    plt.figure(figsize=(15, 10))
    plt.plot(epochs, metric, label=f"Training {plot}")
    plt.plot(epochs, val_metric, label=f"Validation {plot}")
    plt.legend()
    plt.show()


# Lets us do the forecast on the prediction
forecast = []
for index, row in valid_df.iterrows():
    data = row.values[np.newaxis]
    prediction = model.predict(data)
    forecast.append(prediction[0][0])


original = norm_valid_y1[-50:]
forecast = forecast[-50:]

x_axis = range(len(original))

plt.figure(figsize=(15, 10))
plt.plot(x_axis, original, label="Original Data")
plt.plot(x_axis, forecast, label="Prediction Data")
plt.legend()
plt.title("Forecast Plot")
plt.show()


model_mae = tf.keras.losses.mean_absolute_error(original, forecast)
print(f"Final Model MAE: {model_mae}")


# Final MAE with normalisation = 0.11
# Final MAE without normalisation = 2.04

# Conclusion is to try and examine the forecast and the metric and use the one that suit better for the data that is
# dealt with.
