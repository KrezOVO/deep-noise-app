from flask import Flask, render_template, request, jsonify
import plotly
import json
from deepnoiseapp import DeepNoiseApp
import numpy as np

app = Flask(__name__)
app.debug = True
deep_noise_app = DeepNoiseApp(app)

def stairs(x,y):
    X = []
    Y = []
    X.append(x[0])
    X.append(x[0])
    Y.append(y[0])
    Y.append(y[0])
    for i in range(1, len(x)):
        X.append(x[i])
        X.append(x[i])
        Y.append(y[i-1])
        Y.append(y[i])
    return X, Y

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        Qv = float(request.form.get('Qv'))
        DP = float(request.form.get('DP'))
        RPM = float(request.form.get('RPM'))
        type = int(request.form.get('type'))
        method = request.form.get('mode')
        pred, pred_fft = deep_noise_app.predict([Qv, DP, RPM, type])
        y = np.array(range(0,5000,64))
        s_x, s_y = stairs(y, pred_fft)
        graph = dict(
                data=[
                    dict(
                        x=s_x,
                        y=s_y,
                        type='scatter'
                    ),
                ],
                layout=dict(
                    title='三分之一倍频程'
                )
        )
        graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
        res = {
            'db' : pred,
            'fft' : graphJSON
        }
        return jsonify(res)
    else:
        return render_template('layout/index.html')

@app.route('/3octave', methods=['POST'])
def octave_3():

    Qv = float(request.form.get('Qv'))
    DP = float(request.form.get('DP'))
    RPM = float(request.form.get('RPM'))
    type = int(request.form.get('type'))
    method = request.form.get('mode')
    pred_fft = deep_noise_app.predict_3octave([Qv, DP, RPM, type])
    y = np.array(range(0,5000,64))
    s_x, s_y = stairs(y, pred_fft)
    graph = dict(
            data=[
                dict(
                    x=s_x,
                    y=s_y
                ),
            ],
            layout=dict(
                title='三分之一倍频程'
            )
    )
    graphJSON = json.dumps(graph, cls=plotly.utils.PlotlyJSONEncoder)
    res = {
            'db' : 0,
            'fft' : graphJSON
    }
    return jsonify(res)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999)
