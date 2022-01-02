from flask import Flask, render_template, request, url_for,redirect
from model import build_model
import numpy as np


app = Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        predict_model = build_model(training=False)
        zone = float(request.form.get('zone')) / 25000
        room = float(request.form.get('rooms'))
        distance = float(request.form.get('distance'))
        prediction = predict_model.predict(np.array([[zone,room,distance]]))[0] * 1000
        return redirect(url_for('predict', result=prediction))

@app.route('/result/<result>')
def predict(result):
    return f'This House cost {result} dolars approximate'


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001,debug=True)