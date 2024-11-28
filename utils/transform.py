import numpy as np

class Normalizer():
    def __init__(self, mean=[], std=[]):
        self.mean = np.array(mean)
        self.std = np.array(std)
    
    def __call__(self, x):
        return (x-self.mean)/self.std
        
