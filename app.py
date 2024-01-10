# app.py

from flask import Flask, render_template, request
from prediction_script import load_models, make_predictions

app = Flask(__name__)
xg_model, rf_model = load_models()

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

        predictions_xg, predictions_rf = make_predictions(input_data, xg_model, rf_model)

        return render_template("index.html", predictions_xg=predictions_xg, predictions_rf=predictions_rf)

    return render_template("index.html", predictions_xg=None, predictions_rf=None)


if __name__ == '__main__':
    app.run(debug=True)
