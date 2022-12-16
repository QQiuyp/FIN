import torch.nn as nn
from models.Inn import Noise_INN, INN
from block.Haar import HaarDownsampling

class INL(nn.Module):
    def __init__(self):
        super(INL, self).__init__()
        self.model = Noise_INN()
        self.haar = HaarDownsampling(3)

    def forward(self, x, rev=False):

        if not rev:
            out = self.haar(x)
            out = self.model(out)
            out = self.haar(out, rev=True)

        else:
            out = self.haar(x)
            out = self.model(out, rev=True)
            out = self.haar(out, rev=True)

        return out

class FED(nn.Module):
    def __init__(self):
        super(FED, self).__init__()
        self.model = INN()

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)

        else:
            out = self.model(x, rev=True)

        return out
