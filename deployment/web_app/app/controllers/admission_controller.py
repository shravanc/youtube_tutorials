from flask import request, jsonify, render_template, url_for, request, redirect

from app.helpers.prediction import predict_chances

def index():
    if request.method == "POST":
        prediction = predict_chances(request)
        print("prediction--->", prediction)
        return render_template('show.html', prediction=prediction)

    return render_template('index.html')
