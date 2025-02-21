from flask import Flask, render_template, request, jsonify, send_file
import plotly
import json
from deepnoiseapp import DeepNoiseApp
import numpy as np
import pandas as pd
import os
from io import BytesIO

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

def parseInterval(request):
    if request.form.get('QvInterval'):
        QvInterval = int(request.form.get('QvInterval'))
    else:
        QvInterval = 1
    if request.form.get('QvNum'):
        QvNum = int(request.form.get('QvNum'))
    else:
        QvNum = 1
    if request.form.get('DPInterval'):
        DPInterval = int(request.form.get('DPInterval'))
    else:
        DPInterval = 1
    if request.form.get('DPNum'):
        DPNum = int(request.form.get('DPNum'))
    else:
        DPNum = 1
    if request.form.get('RPMInterval'):
        RPMInterval = int(request.form.get('RPMInterval'))
    else:
        RPMInterval = 1
    if request.form.get('RPMNum'):
        RPMNum = int(request.form.get('RPMNum'))
    else:
        RPMNum = 1
    return QvInterval,QvNum,DPInterval,DPNum,RPMInterval,RPMNum

def inputArrange(Qv, DP, RPM):
    data = []
    for i in range(len(Qv)):
        for j in range(len(DP)):
            for k in range(len(RPM)):
                data.append([Qv[i], DP[j], RPM[k]])
    return np.array(data)

def dBA_postprocess(Qv, DP, RPM, pred):
    Qv_fig = []
    DP_fig = []
    RPM_fig = []
    for j in range(len(DP)):
        for k in range(len(RPM)):
            Qv_fig.append(
                dict(
                    x = Qv,
                    y = [pred[k+j*len(RPM)+i*len(RPM)*len(DP)] for i in range(len(Qv))],
                    name=f'DP_{DP[j]}_RPM_{RPM[k]}',
                    type='scatter'
                )
            )
    
    for i in range(len(Qv)):
        for k in range(len(RPM)):
            DP_fig.append(
                dict(
                    x = DP,
                    y = [pred[k+j*len(RPM)+i*len(RPM)*len(DP)] for j in range(len(DP))],
                    name=f'Qv_{Qv[i]}_RPM_{RPM[k]}',
                    type='scatter'
                )
            )
    
    for i in range(len(Qv)):
        for j in range(len(DP)):
            RPM_fig.append(
                dict(
                    x = RPM,
                    y = [pred[k+j*len(RPM)+i*len(RPM)*len(DP)] for k in range(len(RPM))],
                    name=f'Qv_{Qv[i]}_DP_{DP[j]}',
                    type='scatter'
                )
            )
    return Qv_fig, DP_fig, RPM_fig

def octave3_postprocess(Qv, DP, RPM, pred_fft):
    fft_array = np.array(pred_fft)
    # 定义1/3倍频程中心频率
    frequencies = [31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 
                  630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 
                  6300, 8000]
    fig = []
    for i in range(len(Qv)):
        for j in range(len(DP)):
            for k in range(len(RPM)):
                s_x, s_y = stairs(frequencies, pred_fft[k+j*len(RPM)+i*len(RPM)*len(DP)])
                fig.append(
                    dict(x = s_x, y = s_y, name=f'Qv_{Qv[i]}_DP_{DP[j]}_RPM_{RPM[k]}', type='scatter')
                )
    return fig

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        Qv = int(request.form.get('Qv'))
        DP = int(request.form.get('DP'))
        RPM = int(request.form.get('RPM'))
        type = int(request.form.get('type'))
        method = request.form.get('mode')
        QvInterval,QvNum,DPInterval,DPNum,RPMInterval,RPMNum = parseInterval(request)
        Qv = np.array(range(Qv, Qv+QvInterval*QvNum, QvInterval))
        DP = np.array(range(DP, DP+DPInterval*DPNum, DPInterval))
        RPM = np.array(range(RPM, RPM+RPMInterval*RPMNum, RPMInterval))
        data = inputArrange(Qv, DP, RPM)
        input = {'data': data, 'type': type, 'method': method}
        pred = deep_noise_app.predict(input)
        Qv_fig, DP_fig, RPM_fig = dBA_postprocess(Qv, DP, RPM, pred)
        graphs = [
            dict(
                data=Qv_fig,
                layout=dict(
                    title='dBA level predict',
                    xaxis=dict(
                        title=dict(
                            text='Qv'
                        )
                    ),
                    yaxis=dict(
                        title=dict(
                            text='dBA'
                        )
                    ),
                )
            ),
            dict(
                data=DP_fig,
                layout=dict(
                    title='dBA level predict',
                    xaxis=dict(
                        title=dict(
                            text='DP'
                        )
                    ),
                    yaxis=dict(
                        title=dict(
                            text='dBA'
                        )
                    ),
                )
            ),
            dict(
                data=RPM_fig,
                layout=dict(
                    title='dBA level predict',
                    xaxis=dict(
                        title=dict(
                            text='RPM'
                        )
                    ),
                    yaxis=dict(
                        title=dict(
                            text='dBA'
                        )
                    ),
                )
            )
        ]
        graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
        res = {
            'fig' : graphJSON
        }
        return jsonify(res)
    else:
        return render_template('layout/index.html')

@app.route('/3octave', methods=['POST'])
def octave_3():
    Qv = int(request.form.get('Qv'))
    DP = int(request.form.get('DP'))
    RPM = int(request.form.get('RPM'))
    type = int(request.form.get('type'))
    method = request.form.get('mode')
    QvInterval,QvNum,DPInterval,DPNum,RPMInterval,RPMNum = parseInterval(request)
    Qv = np.array(range(Qv, Qv+QvInterval*QvNum, QvInterval))
    DP = np.array(range(DP, DP+DPInterval*DPNum, DPInterval))
    RPM = np.array(range(RPM, RPM+RPMInterval*RPMNum, RPMInterval))
    data = inputArrange(Qv, DP, RPM)
    input = {'data': data, 'type': type, 'method': method}
    pred_fft = deep_noise_app.predict_3octave(input)
    octave_fig = octave3_postprocess(Qv, DP, RPM, pred_fft)
    graphs = [
        dict(
            data=octave_fig,
            layout=dict(
                title='1/3 octave',
                xaxis=dict(
                    title=dict(
                        text='Hz'
                    )
                ),
                yaxis=dict(
                    title=dict(
                        text='dBA'
                    )
                )
            )
    )]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    res = {
            'fig' : graphJSON
    }
    return jsonify(res)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999)
