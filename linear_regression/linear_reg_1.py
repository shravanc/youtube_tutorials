import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load Data to Pandas DataFrame
url = 'https://raw.githubusercontent.com/shravanc/datasets/master/graduate_admission.csv'
df = pd.read_csv(url)
print(df.head())

TARGET = 'admit'

# Categorical Columns
categorical_columns = ['research']
df = pd.get_dummies(df, columns=categorical_columns)
print(df.head())


# Separate Data into Train and Validation Data
train_df = df.sample(frac=0.85, random_state=0)
valid_df = df.drop(train_df.index)


# Removing labels from data
train_labels = train_df.pop(TARGET).values
valid_labels = valid_df.pop(TARGET).values


# Scaling data with train_df
stats = train_df.describe().transpose()
norm_train_df = (train_df-stats['mean'])/stats['std']
norm_valid_df = (valid_df-stats['mean'])/stats['std']

# Storing data from testing prediction
prediction_data = np.array(norm_train_df.iloc[0].values)[np.newaxis]
result = train_labels[0]

# Converting data to numpy
train_data = norm_train_df.to_numpy()
valid_data = norm_valid_df.to_numpy()


# Prepare tf.data.Dataset for training and better performance
def prepare_dataset(data, labels, batch, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


# Preparing Dataset for training
batch_size = 32
buffer = 100
train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)


# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
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
    epochs=10,
    validation_data=valid_dataset
)

# plotting Training and validation Loss and MAE
mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(len(mae))

plt.figure(figsize=(15,10))
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


prediction = model.predict(prediction_data)
print(f"Expected Result: {result}, Prediction Result: {prediction}")

