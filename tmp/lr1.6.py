import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

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
norm_train_df = (train_df-stats['mean'])/stats['std']
norm_valid_df = (valid_df-stats['mean'])/stats['std']
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


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=[len(prediction_data)]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(
    loss=tf.keras.losses.mse,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['mae']
)

history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=valid_dataset
)


mae = history.history['mae']
val_mae = history.history['val_mae']

epochs = range(len(mae))

plt.figure(figsize=(15, 10))
plt.plot(epochs, mae, label=['Training MAE'])
plt.plot(epochs, val_mae, label=['Validation MAE'])
plt.legend()
plt.show()

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(15, 10))
plt.plot(epochs, loss, label=['Training Loss'])
plt.plot(epochs, val_loss, label=['Validation Loss'])
plt.legend()
plt.show()

prediction_data = np.array(prediction_data) #[np.newaxis]
print("---1", prediction_data)
prediction_data = ((prediction_data - stats['mean'])/stats['std']).values
print("---2", prediction_data)
prediction_data = prediction_data[np.newaxis]
print("---3", prediction_data)
prediction = model.predict(prediction_data[np.newaxis])
print(f"Expected Result: {result}, Prediction: {prediction[0][0]}")