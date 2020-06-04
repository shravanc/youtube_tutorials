import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#from scipy import stats

url = 'https://raw.githubusercontent.com/shravanc/datasets/master/graduate_admission.csv'
df = pd.read_csv(url)
df = df.drop('index', axis=1)
print(df.head())

categorical_columns = ['uni_rating', 'research']
df = pd.get_dummies(df, columns=categorical_columns)
raw_df = df.copy()
print(df.head())


def norm(feature):
    mean = df[feature].mean()
    std = df[feature].std()
    df[feature] = (df[feature] - mean) / std


numerical_columns = ['gre', 'toefl', 'sop', 'lor', 'cgpa']
for cat in numerical_columns:
    norm(cat)

print(df.head())


stats = raw_df.describe()
stats = stats.transpose()

print(stats['mean'])

raw_df = (raw_df /stats['mean']) / stats['std']
print(raw_df.head())