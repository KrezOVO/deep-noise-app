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
    
    def predict(self,data):
        transformations = Normalizer(mean=[354.16, 32.17, 2649.37], std=[187.5, 647.17, 2045.62])
        for i in range(5-len(data)):
            data.append(1)
        input = transformations(data[:3])
        type = data[3]
        method = data[4]
        input = torch.tensor(input).to(torch.float32)
        input = input.unsqueeze(0)
        type = torch.LongTensor(type)
        type = type.unsqueeze(0)
        pred = self.model(input, type)

        type_ = torch.LongTensor(np.array(range(data[3]*self.fft_out, (data[3]+1)*self.fft_out)))
        type_ = type_.unsqueeze(0)
        fft_pred = self.fft_model(input, type_)
        pred = pred.squeeze().item()
        fft_pred = fft_pred.squeeze().tolist()
        return pred, fft_pred
