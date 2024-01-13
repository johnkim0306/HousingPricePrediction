# app.py

from flask import Flask, render_template, request
from prediction_script import load_models, make_predictions


app = Flask(__name__)
rf_model = load_models()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_data = {
            'SquareFeet': float(request.form['SquareFeet']),
            'Nbed': int(request.form['Nbed']),
            'Nbath': int(request.form['Nbath']),
            'Neighborhood': request.form['Neighborhood'],
            'YearBuilt': int(request.form['YearBuilt'])
        }

        print(input_data)

        predictions_rf = make_predictions(input_data, rf_model)
        print('app.py: ', predictions_rf)

        return render_template("index.html", predictions_rf=predictions_rf)

    return render_template("index.html", predictions_rf=None)