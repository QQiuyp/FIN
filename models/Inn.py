from torch import nn
from block.InnBlock import Noise_INN_block, INN_block

class Noise_INN(nn.Module):
    def __init__(self):
        super(Noise_INN, self).__init__()

        self.inv1 = Noise_INN_block()
        self.inv2 = Noise_INN_block()
        self.inv3 = Noise_INN_block()
        self.inv4 = Noise_INN_block()
        self.inv5 = Noise_INN_block()
        self.inv6 = Noise_INN_block()
        self.inv7 = Noise_INN_block()
        self.inv8 = Noise_INN_block()

    def forward(self, x, rev=False):

        if not rev:
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)
        else:

            out = self.inv8(x, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)
        return out


class INN(nn.Module):
    def __init__(self):
        super(INN, self).__init__(diff=False,length=30)
        self.diff = diff
        self.inv1 = INN_block()
        self.inv2 = INN_block()
        self.inv3 = INN_block()
        self.inv4 = INN_block()
        self.inv5 = INN_block()
        self.inv6 = INN_block()
        self.inv7 = INN_block()
        self.inv8 = INN_block()
        if self.diff:
            self.diff_layer_pre = nn.Linear(length, 64)
            self.leakrelu = nn.LeakyReLU(inplace=True)
            self.diff_layer_post = nn.Linear(64, length)

        

    def forward(self, x, rev=False):
        if not rev:
            if self.diff:
                x1, x2 = x[0], x[1]
                x2 = self.leakrelu(self.diff_layer_pre(x2))
                x = [x1, x2]
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)
        else:
            out = self.inv8(x, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)
            if self.diff:
                x1, x2 = out
                x2 = self.diff_layer_post(x2)
                out = [x1, x2]
        return out
