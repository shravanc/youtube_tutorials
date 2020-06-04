import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
#import

# Load Data
url = 'https://raw.githubusercontent.com/shravanc/datasets/master/graduate_admission.csv'
df = pd.read_csv(url)
print(df.head())

# Removing Index Column
df = df.drop('index', axis=1)
print(df.head())

# Listing Categorical columns and one hot encoding the column
categorical_column = ['research']
df = pd.get_dummies(df, columns=categorical_column)
print(df.head())

# Separate Data into Train and Valid
train_df = df.sample(frac=0.8, random_state=0)
valid_df = df.drop(train_df.index)

# Separating the labels
TARGET = 'admit'
train_labels = train_df.pop(TARGET).values
valid_labels = valid_df.pop(TARGET).values


# Storing first value in the train for prediction
first_column = train_df.iloc[0]
expected_val = train_labels[0]


# Scaling Data with train data
stats = train_df.describe().transpose()
norm_train_data = (train_df-stats['mean'])/stats['std']
norm_valid_data = (valid_df-stats['mean'])/stats['std']
print(norm_train_data.head())

# Converting to numpy arrays
train_data = norm_train_data.to_numpy()
valid_data = norm_train_data.to_numpy()


# Converting numpy to tf.data.Dataset with defined batches
def prepare_dataset(data, labels, batch, shuffle_buffer):
    dataset = tf.data.Dataset.from_tensor_slices((data, labels))
    dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.batch(batch).prefetch(1)
    return dataset


# hyperparameters
batch_size = 32
buffer = 100


# Prepare Dataset for Training and Validation
train_dataset = prepare_dataset(train_data, train_labels, batch_size, buffer)
valid_dataset = prepare_dataset(valid_data, valid_labels, batch_size, buffer)


model =