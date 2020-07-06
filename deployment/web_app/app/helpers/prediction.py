import tensorflow as tf
import requests
import pandas as pd
import numpy as np

NUM_OF_RESEARCH_CLASSES = 2
URL = 'http://localhost:8501/v1/models/regression_model:predict'

HEADERS = {'Content-Type': 'application/json', 'Accept': 'application/json'}

STATS_FILE = '/tmp/stats.csv'


def get_categories(value):
    return tf.keras.utils.to_categorical([value], num_classes=NUM_OF_RESEARCH_CLASSES)[0].tolist()


def get_params(request):
    params = []

    fields = ['gre', 'toefl', 'uni_rating', 'sop', 'lor', 'cgpa']
    categorical_fields = ['research']

    for field in fields:
        value = request.form.get(field)
        if not value:
            value = 0.
        params.append(float(value))

    for cat_field in categorical_fields:
        value = request.form.get(cat_field)
        if not value:
            value = 0.
        params.extend(get_categories(value))

    stats = pd.read_csv(STATS_FILE)
    params = np.array(params)
    params = (params-stats['mean'])/stats['std']
    return params.tolist()


def get_predictions(params):
    data = {'instances': [params]}
    response = requests.post(URL, json=data, headers=HEADERS)
    return response


def predict_chances(request):
    params = get_params(request)
    print(params)
    predictions = get_predictions(params)
    return predictions.json()['predictions'][0][0]
