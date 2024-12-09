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

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        Qv = float(request.form.get('Qv'))
        DP = float(request.form.get('DP'))
        RPM = float(request.form.get('RPM'))
        type = request.form.get('type')
        method = request.form.get('mode')
        pred, pred_fft = deep_noise_app.predict([Qv, DP, RPM])
        y = np.array(range(0,5000,64))
        s_x, s_y = stairs(y, pred_fft)
        
        # 保存数据用于下载
        app.pred_data = {
            'frequency': s_x,
            'amplitude': s_y
        }
        
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

@app.route('/download_data')
def download_data():
    try:
        # 创建DataFrame
        df = pd.DataFrame(app.pred_data)
        
        # 创建BytesIO对象
        excel_buffer = BytesIO()
        
        # 将DataFrame保存到Excel
        with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='预测数据', index=False)
        
        excel_buffer.seek(0)
        
        return send_file(
            excel_buffer,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name='prediction_data.xlsx'
        )
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9999)
