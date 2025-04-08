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

    if request.form.get('wheelDiameterInterval'):
        wheelDiameterInterval = float(request.form.get('wheelDiameterInterval'))
    else:
        wheelDiameterInterval = 1.0
    if request.form.get('wheelDiameterNum'):
        wheelDiameterNum = int(request.form.get('wheelDiameterNum'))
    else:
        wheelDiameterNum = 1

    if request.form.get('wheelHeightInterval'):
        wheelHeightInterval = float(request.form.get('wheelHeightInterval'))
    else:
        wheelHeightInterval = 1.0
    if request.form.get('wheelHeightNum'):
        wheelHeightNum = int(request.form.get('wheelHeightNum'))
    else:
        wheelHeightNum = 1

    if request.form.get('wheelHoleDiameterInterval'):
        wheelHoleDiameterInterval = float(request.form.get('wheelHoleDiameterInterval'))
    else:
        wheelHoleDiameterInterval = 1.0
    if request.form.get('wheelHoleDiameterNum'):
        wheelHoleDiameterNum = int(request.form.get('wheelHoleDiameterNum'))
    else:
        wheelHoleDiameterNum = 1

    return (QvInterval, QvNum, DPInterval, DPNum, RPMInterval, RPMNum,
            wheelDiameterInterval, wheelDiameterNum, wheelHeightInterval, wheelHeightNum,
            wheelHoleDiameterInterval, wheelHoleDiameterNum)

def inputArrange(Qv, DP, RPM, wheelDiameters, wheelHeights, wheelHoleDiameters):
    data = []
    for i in range(len(Qv)):
        for j in range(len(DP)):
            for k in range(len(RPM)):
                for wd in wheelDiameters:
                    for wh in wheelHeights:
                        for whd in wheelHoleDiameters:
                            data.append([Qv[i], DP[j], RPM[k], wd, wh, whd])
    return np.array(data)

def dBA_postprocess(Qv, DP, RPM, wheelDiameters, wheelHeights, wheelHoleDiameters, pred):
    Qv_fig = []
    DP_fig = []
    RPM_fig = []
    WD_fig = []
    WH_fig = []
    WHD_fig = []

    for j in range(len(DP)):
        for k in range(len(RPM)):
            for wd in range(len(wheelDiameters)):
                for wh in range(len(wheelHeights)):
                    for whd in range(len(wheelHoleDiameters)):
                        Qv_fig.append(
                            dict(
                                x=Qv,
                                y=[pred[whd + len(wheelHoleDiameters) * (wh + len(wheelHeights) * (wd + len(wheelDiameters) * (k + len(RPM) * (j + len(DP) * i))))] for i in range(len(Qv))],
                                name=f'DP_{DP[j]}_RPM_{RPM[k]}_WD_{wheelDiameters[wd]}_WH_{wheelHeights[wh]}_WHD_{wheelHoleDiameters[whd]}',
                                type='scatter'
                            )
                        )

    for i in range(len(Qv)):
        for k in range(len(RPM)):
            for wd in range(len(wheelDiameters)):
                for wh in range(len(wheelHeights)):
                    for whd in range(len(wheelHoleDiameters)):
                        DP_fig.append(
                            dict(
                                x=DP,
                                y=[pred[whd + len(wheelHoleDiameters) * (wh + len(wheelHeights) * (wd + len(wheelDiameters) * (k + len(RPM) * (j + len(DP) * i))))] for j in range(len(DP))],
                                name=f'Qv_{Qv[i]}_RPM_{RPM[k]}_WD_{wheelDiameters[wd]}_WH_{wheelHeights[wh]}_WHD_{wheelHoleDiameters[whd]}',
                                type='scatter'
                            )
                        )

    for i in range(len(Qv)):
        for j in range(len(DP)):
            for wd in range(len(wheelDiameters)):
                for wh in range(len(wheelHeights)):
                    for whd in range(len(wheelHoleDiameters)):
                        RPM_fig.append(
                            dict(
                                x=RPM,
                                y=[pred[whd + len(wheelHoleDiameters) * (wh + len(wheelHeights) * (wd + len(wheelDiameters) * (k + len(RPM) * (j + len(DP) * i))))] for k in range(len(RPM))],
                                name=f'Qv_{Qv[i]}_DP_{DP[j]}_WD_{wheelDiameters[wd]}_WH_{wheelHeights[wh]}_WHD_{wheelHoleDiameters[whd]}',
                                type='scatter'
                            )
                        )

    for i in range(len(Qv)):
        for j in range(len(DP)):
            for k in range(len(RPM)):
                for wh in range(len(wheelHeights)):
                    for whd in range(len(wheelHoleDiameters)):
                        WD_fig.append(
                            dict(
                                x=wheelDiameters,
                                y=[pred[whd + len(wheelHoleDiameters) * (wh + len(wheelHeights) * (wd + len(wheelDiameters) * (k + len(RPM) * (j + len(DP) * i))))] for wd in range(len(wheelDiameters))],
                                name=f'Qv_{Qv[i]}_DP_{DP[j]}_RPM_{RPM[k]}_WH_{wheelHeights[wh]}_WHD_{wheelHoleDiameters[whd]}',
                                type='scatter'
                            )
                        )

    for i in range(len(Qv)):
        for j in range(len(DP)):
            for k in range(len(RPM)):
                for wd in range(len(wheelDiameters)):
                    for whd in range(len(wheelHoleDiameters)):
                        WH_fig.append(
                            dict(
                                x=wheelHeights,
                                y=[pred[whd + len(wheelHoleDiameters) * (wh + len(wheelHeights) * (wd + len(wheelDiameters) * (k + len(RPM) * (j + len(DP) * i))))] for wh in range(len(wheelHeights))],
                                name=f'Qv_{Qv[i]}_DP_{DP[j]}_RPM_{RPM[k]}_WD_{wheelDiameters[wd]}_WHD_{wheelHoleDiameters[whd]}',
                                type='scatter'
                            )
                        )

    for i in range(len(Qv)):
        for j in range(len(DP)):
            for k in range(len(RPM)):
                for wd in range(len(wheelDiameters)):
                    for wh in range(len(wheelHeights)):
                        WHD_fig.append(
                            dict(
                                x=wheelHoleDiameters,
                                y=[pred[whd + len(wheelHoleDiameters) * (wh + len(wheelHeights) * (wd + len(wheelDiameters) * (k + len(RPM) * (j + len(DP) * i))))] for whd in range(len(wheelHoleDiameters))],
                                name=f'Qv_{Qv[i]}_DP_{DP[j]}_RPM_{RPM[k]}_WD_{wheelDiameters[wd]}_WH_{wheelHeights[wh]}',
                                type='scatter'
                            )
                        )

    return Qv_fig, DP_fig, RPM_fig, WD_fig, WH_fig, WHD_fig

def octave3_postprocess(Qv, DP, RPM, wheelDiameters,wheelHeights,wheelHoleDiameters, pred_fft):
    fft_array = np.array(pred_fft)
    # 定义1/3倍频程中心频率
    frequencies = [31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 
                  630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 
                  6300, 8000, 10000]
    fig = []
    for i in range(len(Qv)):
        for j in range(len(DP)):
            for k in range(len(RPM)):
                for wd in range(len(wheelDiameters)):
                    for wh in range(len(wheelHeights)):
                        for whd in range(len(wheelHoleDiameters)):
                            index = (whd + len(wheelHoleDiameters) * (wh + len(wheelHeights) * (wd + len(wheelDiameters) * (k + len(RPM) * (j + len(DP) * i)))))
                            s_x, s_y = stairs(frequencies, pred_fft[index])
                            fig.append(
                                dict(
                                    x=s_x,
                                    y=s_y,
                                    name=f'{Qv[i]}_{DP[j]}_{RPM[k]}_{wheelDiameters[wd]}_{wheelHeights[wh]}_{wheelHoleDiameters[whd]}',
                                    type='scatter'
                                )
                            )
    layout = dict(
        title='1/3 octave',
        xaxis=dict(
            title=dict(
                text='Frequency (Hz)'
            )
        ),
        yaxis=dict(
            title=dict(
                text='dBA'
            )
        )
    )
    return fig, layout

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        Qv = int(request.form.get('Qv', 0))
        DP = int(request.form.get('DP', 0))
        RPM = int(request.form.get('RPM', 0))
        wheelDiameter = float(request.form.get('wheelDiameter', 0))
        wheelHeight = float(request.form.get('wheelHeight', 0))
        wheelHoleDiameter = float(request.form.get('wheelHoleDiameter', 0))
        method = int(request.form.get('mode', 0))
        bowl = int(request.form.get('bowl', 0))

        QvInterval, QvNum, DPInterval, DPNum, RPMInterval, RPMNum, wheelDiameterInterval, wheelDiameterNum, wheelHeightInterval, wheelHeightNum, wheelHoleDiameterInterval, wheelHoleDiameterNum = parseInterval(request)
        Qv = np.array(range(Qv, Qv + QvInterval * QvNum, QvInterval))
        DP = np.array(range(DP, DP + DPInterval * DPNum, DPInterval))
        RPM = np.array(range(RPM, RPM + RPMInterval * RPMNum, RPMInterval))
        wheelDiameters = np.arange(wheelDiameter, wheelDiameter + wheelDiameterInterval * wheelDiameterNum, wheelDiameterInterval)
        wheelHeights = np.arange(wheelHeight, wheelHeight + wheelHeightInterval * wheelHeightNum, wheelHeightInterval)
        wheelHoleDiameters = np.arange(wheelHoleDiameter, wheelHoleDiameter + wheelHoleDiameterInterval * wheelHoleDiameterNum, wheelHoleDiameterInterval)
        
        data = inputArrange(Qv, DP, RPM, wheelDiameters, wheelHeights, wheelHoleDiameters)
        
        input = {'data': data, 'method': method, 'bowl': bowl}
        pred = deep_noise_app.predict(input)
        Qv_fig, DP_fig, RPM_fig, WD_fig, WH_fig, WHD_fig = dBA_postprocess(Qv, DP, RPM, wheelDiameters, wheelHeights, wheelHoleDiameters, pred)
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
                    legend=dict(
                        orientation='h',
                        x=0,
                        y=-0.3
                    ),
                    margin=dict(b=100)
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
                    legend=dict(
                        orientation='h',
                        x=0,
                        y=-0.3
                    ),
                    margin=dict(b=100)
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
                    legend=dict(
                        orientation='h',
                        x=0,
                        y=-0.3
                    ),
                    margin=dict(b=100)
                )
            ),
            dict(
                data=WD_fig,
                layout=dict(
                    title='dBA level predict',
                    xaxis=dict(
                        title=dict(
                            text='WD'
                        )
                    ),
                    yaxis=dict(
                        title=dict(
                            text='dBA'
                        )
                    ),
                    legend=dict(
                        orientation='h',
                        x=0,
                        y=-0.3
                    ),
                    margin=dict(b=100)
                )
            ),
            dict(
                data=WH_fig,
                layout=dict(
                    title='dBA level predict',
                    xaxis=dict(
                        title=dict(
                            text='WH'
                        )
                    ),
                    yaxis=dict(
                        title=dict(
                            text='dBA'
                        )
                    ),
                    legend=dict(
                        orientation='h',
                        x=0,
                        y=-0.3
                    ),
                    margin=dict(b=100)
                )
            ),
            dict(
                data=WHD_fig,
                layout=dict(
                    title='dBA level predict',
                    xaxis=dict(
                        title=dict(
                            text='WHD'
                        )
                    ),
                    yaxis=dict(
                        title=dict(
                            text='dBA'
                        )
                    ),
                    legend=dict(
                        orientation='h',
                        x=0,
                        y=-0.3
                    ),
                    margin=dict(b=100)
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
    Qv = int(request.form.get('Qv', 0))
    DP = int(request.form.get('DP', 0))
    RPM = int(request.form.get('RPM', 0))
    wheelDiameter = float(request.form.get('wheelDiameter', 0))
    wheelHeight = float(request.form.get('wheelHeight', 0))
    wheelHoleDiameter = float(request.form.get('wheelHoleDiameter', 0))
    method = int(request.form.get('mode', 0))
    bowl = int(request.form.get('bowl', 0))

    QvInterval, QvNum, DPInterval, DPNum, RPMInterval, RPMNum, wheelDiameterInterval, wheelDiameterNum, wheelHeightInterval, wheelHeightNum, wheelHoleDiameterInterval, wheelHoleDiameterNum = parseInterval(request)
    Qv = np.array(range(Qv, Qv + QvInterval * QvNum, QvInterval))
    DP = np.array(range(DP, DP + DPInterval * DPNum, DPInterval))
    RPM = np.array(range(RPM, RPM + RPMInterval * RPMNum, RPMInterval))
    wheelDiameters = np.arange(wheelDiameter, wheelDiameter + wheelDiameterInterval * wheelDiameterNum, wheelDiameterInterval)
    wheelHeights = np.arange(wheelHeight, wheelHeight + wheelHeightInterval * wheelHeightNum, wheelHeightInterval)
    wheelHoleDiameters = np.arange(wheelHoleDiameter, wheelHoleDiameter + wheelHoleDiameterInterval * wheelHoleDiameterNum, wheelHoleDiameterInterval)

    data = inputArrange(Qv, DP, RPM, wheelDiameters, wheelHeights, wheelHoleDiameters)

    input = {'data': data, 'method': method, 'bowl': bowl}
    pred_fft = deep_noise_app.predict_3octave(input)
    octave_fig, layout = octave3_postprocess(Qv, DP, RPM, wheelDiameters,wheelHeights,wheelHoleDiameters, pred_fft)
    graphs = [
        dict(
            data=octave_fig,
            layout=layout
        )
    ]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    res = {
        'fig': graphJSON
    }
    return jsonify(res)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999)
