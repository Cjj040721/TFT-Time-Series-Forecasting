import os
import torch
import numpy as np

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None
    
    def _acquire_device(self):
        if self.args.use_gpu and torch.cuda.is_available():
            device = torch.device(f"cuda:{self.args.gpu}")
            print(f"Use GPU: {device}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Use GPU: mps")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
    