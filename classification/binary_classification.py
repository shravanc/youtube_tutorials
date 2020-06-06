import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# load data
url = 'https://raw.githubusercontent.com/shravanc/datasets/master/heart.csv'
df = pd.read_csv(url)
print(df.head())

TARGET = 'target'

# Categorical Columns
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'ca', 'thal']
df = pd.get_dummies(df, columns=categorical_columns)
print(df.head())

# Separating to Train and Valid
train_df = df.sample(frac=0.85, random_state=0)
valid_df = df.drop(train_df.index)

train_labels = train_df.pop(TARGET).values
valid_labels = valid_df.pop(TARGET).values

# Scaling
stats = train_df.describe().transpose()
train_df = (train_df - stats['mean']) / stats['std']
valid_df = (valid_df - stats['mean']) / stats['std']
print(train_df.head())

# storing data for future prediction
prediction_data_0 = np.array(train_df.iloc[0])[np.newaxis]
result_0 = train_labels[0]
prediction_data_1 = np.array(train_df.iloc[1])[np.newaxis]
result_1 = train_labels[1]

# Converting to numpy
train_data = train_df.to_numpy()
valid_data = valid_df.to_numpy()


# Converting numpy to tf dataset
def prepare_dataset(data, label, batch, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, label))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


# Preparing dataset
batch_size = 32
buffer = 50
train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)

# Building Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.binary_crossentropy,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['accuracy', 'mae']
)

history = model.fit(
    train_dataset,
    epochs=100,
    validation_data=valid_dataset
)

# Plotting Metrics Curve
plots = ['accuracy', 'mae', 'loss']
for plot in plots:
    metric = history.history[plot]
    val_metric = history.history[f"val_{plot}"]

    epochs = range(len(metric))

    plt.figure(figsize=(15, 10))
    plt.plot(epochs, metric, label=f"Training {plot}")
    plt.plot(epochs, val_metric, label=f"Validation {plot}")
    plt.legend()
    plt.title(f"Training and Validation for {plot}")
    plt.show()

prediction_0 = model.predict(prediction_data_0)
prediction_1 = model.predict(prediction_data_1)
print(f"Expected Result: {result_0}, Prediction Result: {prediction_0}")
print(f"Expected Result: {result_1}, Prediction Result: {prediction_1}")
