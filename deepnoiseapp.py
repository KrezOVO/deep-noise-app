from model.nonlinear import NonLinear, NonLinearType, NonLinearTypeBin, NonLinearTypeBinModel, NonLinearTypeModel, NonLinearBowlBinMode, NonLinearBowlMode
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
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = NonLinearBowlMode(nc=config['model_nc'], out_nc=2, num_sheets=4)
        saved_state_dict = torch.load(config['model_path'], map_location=device)
        model.load_state_dict(saved_state_dict)
        model.eval()

        fft_model = NonLinearBowlBinMode(nc=config['fft_nc'], out_nc=2, num_bins=26, num_sheets=4)
        saved_state_dict = torch.load(config['fft_model_path'], map_location=device)
        fft_model.load_state_dict(saved_state_dict)
        fft_model.eval()
        self.model = model
        self.fft_model = fft_model
        self.fft_out = 26
        app.predict = self.predict
    
    def predict(self, data):
        transformations = Normalizer(mean=[363.80, 46.21, 2457.96, 149.38, 67.70, 7.65], std=[125.97, 199.17, 941.75, 5.73, 6.91, 0.10])
        input = transformations(data['data'])
        method = data['method']
        input = torch.tensor(input).to(torch.float32)
        pred = self.model(input)
        bowl_ = data['bowl']
        pred = pred[:, method, bowl_]
        pred = pred.squeeze().tolist()
        if isinstance(pred, float):
            pred = [pred]
        return pred

    def predict_3octave(self,data):
        transformations = Normalizer(mean=[363.80, 46.21, 2457.96, 149.38, 67.70, 7.65], std=[125.97, 199.17, 941.75, 5.73, 6.91, 0.10])
        input = transformations(data['data'])
        method = data['method']
        input = torch.tensor(input).to(torch.float32)
        fft_pred = self.fft_model(input) 
        bowl_ = data['bowl']
        fft_pred = fft_pred[:, method, bowl_, :]
        fft_pred = torch.clamp(fft_pred, min=0.0)
        fft_pred = fft_pred.tolist()
        return fft_pred
