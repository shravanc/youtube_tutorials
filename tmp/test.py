import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

url = 'https://raw.githubusercontent.com/shravanc/datasets/master/graduate_admission.csv'
df = pd.read_csv(url)
print(df.head(2))

df = df.drop(['index'], axis=1)
print(df.head(2))

COLUMN = ['gre', 'toefl', 'uni_rating', 'sop', 'lor', 'cgpa', 'research']
TARGET = 'admit'


train_df = df.sample(frac=0.8)
valid_df = df.drop(train_df.index)

train_labels = train_df.drop(TARGET)
valid_labels = valid_df.drop(TARGET)

train_labels = train_labels.values
valid_labels = valid_labels.values

train_data = train_df.to_numpy()
valid_data = valid_df.to_numpy()
