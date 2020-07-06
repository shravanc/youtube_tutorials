import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import shutil

# Load Data to Pandas DataFrame
url = 'https://raw.githubusercontent.com/shravanc/datasets/master/graduate_admission.csv'
df = pd.read_csv(url)
print(df.head())

TARGET = 'admit'

# Categorical Columns
categorical_columns = ['research']
df = pd.get_dummies(df, columns=categorical_columns)


# Separate Data into Train and Validation Data
train_df = df.sample(frac=0.85, random_state=0)
validation = df.drop(train_df.index)

valid_df = validation.sample(frac=0.9, random_state=0)
test_df = validation.drop(valid_df.index)


# Removing labels from data
train_df.pop('index')
valid_df.pop('index')
test_df.pop('index')
train_labels = train_df.pop(TARGET).values
valid_labels = valid_df.pop(TARGET).values
test_labels = test_df.pop(TARGET).values

# Scaling data with train_df
stats_file = '/tmp/stats.csv'
stats = train_df.describe().transpose()
stats.to_csv(stats_file, index=False)
norm_train_df = (train_df-stats['mean'])/stats['std']
norm_valid_df = (valid_df-stats['mean'])/stats['std']
norm_test_df = (test_df-stats['mean'])/stats['std']

# Storing data from testing prediction
prediction_data = np.array(norm_train_df.iloc[0].values)[np.newaxis]
result = train_labels[0]

# Converting data to numpy
train_data = norm_train_df.to_numpy()
valid_data = norm_valid_df.to_numpy()
test_data = norm_test_df.to_numpy()


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
test_dataset = prepare_dataset(test_data, test_labels, batch_size, buffer)

# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])


# Compiling the Model
model.compile(
    loss=tf.keras.losses.mse,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['mae']
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_mae',
    patience=2,
)

# Training the Model
history = model.fit(
    train_dataset,
    epochs=1000,
    validation_data=valid_dataset,
    callbacks=[early_stopping]
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

evaluation = model.evaluate(test_dataset)
print("Loss and MAE: ", evaluation)


# Without Early Stopping and just 10 epoch:
# Loss and MAE:  [0.024596789851784706, 0.13961200416088104]

# With Early stopping:
# Loss and MAE:  [0.013992768712341785, 0.09679555892944336]

# With Early Stopping and Learning Rate:
# Loss and MAE:  [0.017241602763533592, 0.10332953184843063]


save_path = '/tmp/regression_model/1'
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path)

tf.keras.models.save_model(model, save_path)