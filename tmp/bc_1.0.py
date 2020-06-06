import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


url = 'https://raw.githubusercontent.com/shravanc/datasets/master/heart.csv'
df = pd.read_csv(url)
print(df.head())

# Categorical_column
categorical_columns = ['sex', 'cp', 'restecg', 'ca', 'thal']
df = pd.get_dummies(df, columns=categorical_columns)
print(df.head())

TARGET = 'target'

# Separting the data to Train and Valid
train_df = df.sample(frac=0.8, random_state=0)
valid_df = df.drop(train_df.index)


# remove labels
train_labels = train_df.pop('target').values
valid_labels = valid_df.pop('target').values


# Scaling:
stats = train_df.describe().transpose()
train_df = (train_df/stats['mean'])/stats['std']
valid_df = (valid_df/stats['std'])/stats['std']


# storing for future prediction
prediction_data = np.array(train_df.iloc[0])[np.newaxis]
result = train_labels[0]


# Converting to numpy
train_data = train_df.to_numpy()
valid_data = valid_df.to_numpy()


# Converting to Dataset
def prepare_dataset(data, labels, batch, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


# Prepare dataset
batch_size = 300
buffer = 500
train_dataset = prepare_dataset(train_df, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_df, valid_labels, batch_size, buffer)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.RMSprop(lr=1e-4, momentum=0.9),
    metrics=['accuracy', 'mae']
)

# lr_schedule = tf.keras.callbacks.LearningRateScheduler(
#     lambda epoch: 1e-4 * 10**(epoch/20)
# )

history = model.fit(
    train_dataset,
    epochs=150,
    validation_data=valid_dataset,
    # callbacks=[lr_schedule]
)

# plt.semilogx(history.history["lr"], history.history["loss"])
# plt.axis([1e-8, 1e-4, 0, 30])
# plt.show()

mae = history.history['mae']
val_mae = history.history['val_mae']

epochs = range(len(mae))

plt.figure(figsize=(15, 10))
plt.plot(epochs, mae, label='Training MAE') 
plt.plot(epochs, val_mae, label='Validation MAE')
plt.legend()
plt.show()


prediction = model.predict(prediction_data)
print(f"Expected Result: {result}, Prediction Result: {prediction}")