import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Load Data
url = 'https://raw.githubusercontent.com/shravanc/datasets/master/iris.csv'
df = pd.read_csv(url)
print(df.head())

classes = df['class'].unique()
num_classes = len(classes)
dummy_class = []
mapper = {}
for i, cl in enumerate(classes):
    mapper[cl] = i
    dummy_class.append(f"class_{i}")

df = df.replace(mapper)
df = pd.get_dummies(df, columns=['class'])


# Splitting data to train and valid
train_df = df.sample(frac=0.85, random_state=1)
valid_df = df.drop(train_df.index)


train_labels_df = pd.concat([train_df.pop(dc) for dc in dummy_class], axis=1)
valid_labels_df = pd.concat([valid_df.pop(dc) for dc in dummy_class], axis=1)


# Scaling dataset
stats = train_df.describe().transpose()
train_df = (train_df-stats['mean'])/stats['std']
valid_df = (valid_df-stats['mean'])/stats['std']


# Converting to numpy
train_data = train_df.to_numpy()
valid_data = valid_df.to_numpy()


train_labels = train_labels_df.to_numpy()
valid_labels = valid_labels_df.to_numpy()


# Storing data for future prediction
prediction_data = train_df.iloc[0]
result = train_labels_df.iloc[0]


# Prepare dataset
def prepare_dataset(data, labels, batch, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


# training and valida dataset
batch_size = 32
buffer = 50
train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)


# Build Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])


model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['accuracy', 'mae']
)

history = model.fit(
    train_dataset,
    epochs=400,
    validation_data=valid_dataset
)


# plotting Graphs
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


prediction_data = prediction_data.values[np.newaxis]
prediction = model.predict(prediction_data)
print(f"Expected Result: {np.argmax(result)}, Prediction Result: {np.argmax(prediction)}")


index = 1
prediction_data = train_df.iloc[index]
result = train_labels_df.iloc[index]
prediction_data = prediction_data.values[np.newaxis]
prediction = model.predict(prediction_data)
print(f"Expected Result: {np.argmax(result)}, Prediction Result: {np.argmax(prediction)}")


index = 3
prediction_data = train_df.iloc[index]
result = train_labels_df.iloc[index]
prediction_data = prediction_data.values[np.newaxis]
prediction = model.predict(prediction_data)
print(f"Expected Result: {np.argmax(result)}, Prediction Result: {np.argmax(prediction)}")

index = 5
prediction_data = train_df.iloc[index]
result = train_labels_df.iloc[index]
prediction_data = prediction_data.values[np.newaxis]
prediction = model.predict(prediction_data)
print(f"Expected Result: {np.argmax(result)}, Prediction Result: {np.argmax(prediction)}")

index = 4
prediction_data = train_df.iloc[index]
result = train_labels_df.iloc[index]
prediction_data = prediction_data.values[np.newaxis]
prediction = model.predict(prediction_data)
print(f"Expected Result: {np.argmax(result)}, Prediction Result: {np.argmax(prediction)}")

index = 10
prediction_data = train_df.iloc[index]
result = train_labels_df.iloc[index]
prediction_data = prediction_data.values[np.newaxis]
prediction = model.predict(prediction_data)
print(f"Expected Result: {np.argmax(result)}, Prediction Result: {np.argmax(prediction)}")