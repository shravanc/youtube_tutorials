import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


url = 'https://raw.githubusercontent.com/shravanc/datasets/master/iris.csv'
df = pd.read_csv(url)
print(df.head())

classes = df['class'].unique()
mapper = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
df = df.replace(mapper)
print(df.head())


# splitting to Train and valid
train_df = df.sample(frac=0.8, random_state=1)
valid_df = df.drop(train_df.index)
print(train_df.head())


tr_labels = train_df.pop('class')
train_labels = pd.get_dummies(tr_labels, columns=[0]).values[np.newaxis]

va_labels = valid_df.pop('class')
valid_labels = pd.get_dummies(va_labels, columns=[0]).values[np.newaxis]


# scaling
stats = train_df.describe().transpose()
train_df = (train_df-stats['mean'])/stats['std']
valid_df = (valid_df-stats['mean'])/stats['std']


train_data = np.array(train_df.to_numpy())[np.newaxis]
valid_data = np.array(valid_df.to_numpy())[np.newaxis]


def prepare_dataset(data, labels, batch, shuffle_buffer):
    print(data)
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


batch_size = 32
buffer = 50
train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=[4]),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.SGD(),
    metrics=['accuracy', 'mae']
)


history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=valid_dataset
)


plots = ['accuracy', 'mae', 'loss']
for plot in plots:
    metric = history.history[plot]
    val_metric = history.history[f"val_{plot}"]

    epochs = range(len(metric))

    plt.figure(figsize=(15, 10))
    plt.plot(epochs, metric, label=f"Training {plot}")
    plt.plot(epochs, val_metric, label=f"Validation {plot}")
    plt.legend()
    #plt.show()


prediction_data = np.array(train_df.iloc[0])[np.newaxis]
result = tr_labels.iloc[0]
prediction = model.predict(prediction_data)
print(f"Expected Result: {result}, Prediction Result: {np.argmax(prediction)}")

prediction_data = np.array(train_df.iloc[1])[np.newaxis]
result = tr_labels.iloc[1]
prediction = model.predict(prediction_data)
print(f"Expected Result: {result}, Prediction Result: {np.argmax(prediction)}")

prediction_data = np.array(train_df.iloc[4])[np.newaxis]
result = tr_labels.iloc[4]
prediction = model.predict(prediction_data)
print(f"Expected Result: {result}, Prediction Result: {np.argmax(prediction)}")
