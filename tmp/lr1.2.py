import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


url = 'https://raw.githubusercontent.com/shravanc/datasets/master/graduate_admission.csv'
df = pd.read_csv(url)
df = df.drop('index', axis=1)
print(df.head())


categorical_columns = ['research', 'uni_rating']
df = pd.get_dummies(df, columns=categorical_columns)
print(df.head())

df['gre'].hist(bins=50)
#plt.show()

stats = df.describe().transpose()

df = (df-stats['mean']) / stats['std']
print(df.head())

df['gre'].hist(bins=50)
#plt.show()


train_df = df.sample(frac=0.8, random_state=0)
valid_df = df.drop(train_df.index)

TARGET

train_labels = train_df.pop(TARGET)