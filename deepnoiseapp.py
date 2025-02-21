from model.nonlinear import NonLinear, NonLinearType, NonLinearTypeBin
from utils.transform import Normalizer
import json
import torch
import numpy as np

class DeepNoiseApp:
    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app):
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        model = NonLinearType(nc=config['model_nc'])
        saved_state_dict = torch.load(config['model_path'], weights_only=True)
        model.load_state_dict(saved_state_dict)
        model.eval()

        fft_model = NonLinearTypeBin(nc=config['fft_nc'], out_nc=14, num_bins=config['fft_num_bins'])
        saved_state_dict = torch.load(config['fft_model_path'], weights_only=True)
        fft_model.load_state_dict(saved_state_dict)
        fft_model.eval()
        self.model = model
        self.fft_model = fft_model
        self.fft_out = config['fft_num_bins']
        app.predict = self.predict
    
    def predict(self, data):
        transformations = Normalizer(mean=[354.16, 32.17, 2649.37], std=[187.5, 647.17, 2045.62])
        input = transformations(data['data'])
        method = data['method']
        input = torch.tensor(input).to(torch.float32)
        type_ = torch.LongTensor([data['type'] for _ in range(len(data['data']))])
        type_ = type_.unsqueeze(1)
        pred = self.model(input, type_)
        pred = pred.squeeze().tolist()
        if isinstance(pred, float):
            pred = [pred]
        return pred

    def predict_3octave(self,data):
        transformations = Normalizer(mean=[354.16, 32.17, 2649.37], std=[187.5, 647.17, 2045.62])
        input = transformations(data['data'])
        method = data['method']
        input = torch.tensor(input).to(torch.float32)
        type_ = torch.LongTensor(np.array([np.array(range(data['type']*self.fft_out, (data['type']+1)*self.fft_out)) for _ in range(len(data['data']))]))
        fft_pred = self.fft_model(input, type_)
        fft_pred = fft_pred.tolist()
        return fft_pred
