import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================Load Data========================
url = 'https://raw.githubusercontent.com/shravanc/datasets/master/automobile.csv'
df = pd.read_csv(url)
print(df.head())

# =====================Load Data========================

# Categorical Columns
categorrical_columns = ['make', 'fuel_type', 'aspiration', 'doors', 'body_type', 'drive_wheels', 'engine_location',
                        'engine_type', 'cylinders', 'fuel_system', '']

# Numeric Columns
numeric_columns = ['symboling', 'n_loss', 'wheel_base', 'length', 'width', 'height', 'curb_weight', 'engine_size',
                   'bore', 'stroke', 'compression_ratio', 'horsepower', 'peak_rpm', 'city_mpg', 'highway_mpg']

# Numeric Yet Categorical
numeric_yet_categorical_columns = []


# TARGET
TARGET = 'price'

# print(df.loc[:, df.dtypes == float].columns)
# print(df.loc[:, df.dtypes == int].columns)

df = df.replace('?', np.NAN)
print(df.isna().sum())

# Missing Data Columns
missing_data_columns = ['n_loss', 'doors', 'bore', 'stroke', 'horsepower', 'peak_rpm', 'price']

