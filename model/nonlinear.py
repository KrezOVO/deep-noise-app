import torch
from torch import nn

class NonLinear(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=1):
        super(NonLinear, self).__init__()

        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc)


    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        output = self.out(x)

        return output
    

class NonLinearType(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=18):
        super(NonLinearType, self).__init__()

        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc)


    def forward(self, inp, type_):
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.gather(1, type_)
        return output

class NonLinearTypeBin(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=18, num_bins=51):
        super(NonLinearTypeBin, self).__init__()
        self.num_bins = num_bins
        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc*num_bins)

    def forward(self, inp, type_):
        x = self.relu(self.hidden(inp))
        out = self.out(x)
        output = out.gather(1, type_)
        return output
    
class NonLinearMultiBin(nn.Module):
    def __init__(self, in_nc=3, nc=1600, out_nc=1):
        super(NonLinearMultiBin, self).__init__()

        self.hidden = nn.Linear(in_nc, nc)
        self.relu = nn.LeakyReLU()
        self.out = nn.Linear(nc, out_nc)
        self.out0 = nn.Linear(nc, 18)
        self.out1 = nn.Linear(nc, 6)
        self.out2 = nn.Linear(nc, 4)


    def forward(self, inp):
        x = self.relu(self.hidden(inp))
        output = self.out(x)
        output0 = self.out0(x)
        output1 = self.out1(x)
        output2 = self.out2(x)
        return output, output0, output1, output2