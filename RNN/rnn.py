import torch
import torch.nn as nn
import  torch.nn.functional as F


__author__ = "__Girish_Hegde__"


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh'):
        super().__init__()
        # h = act(wh.h + bh + wx.x + bx)
        self.i2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.h2o = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.nonlinearity = getattr(F, nonlinearity)

    def forward(self, input, hidden):
        h = self.i2h(input) + self.h2h(hidden)
        o = self.h2o(h)
        return o, self.nonlinearity(h)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, ):
        super().__init__()

        rnns = [RNNCell(input_size, hidden_size)]  + \
            [RNNCell(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        self.rnns = nn.ModuleList(rnns)
        self.hidden_size = hidden_size

    def forward(self, x):
        hidden = self.init_hidden(x.shape[1])
        # out = []
        out = torch.zeros(x.shape[0], x.shape[1], self.hidden_size)
        for timestep in range(x.shape[0]):
            output = x[timestep]
            for rnn_layer in self.rnns:
                output, hidden = rnn_layer(output, hidden)
            out[timestep] = output
            # out.append(output)
        # out = torch.stack(out)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


if __name__ == '__main__':
    bs = 4
    timesteps = 3
    num_layers = 2
    input_size = 10
    hidden_size = 20

    net = RNN(input_size, hidden_size, num_layers)
    print(net)

    inp = torch.randn(timesteps, bs, input_size)
    out, hdn = net(inp)
    print(out.shape)
