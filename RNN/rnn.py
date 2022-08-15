import torch
import torch.nn as nn


__author__ = "__Girish_Hegde__"


class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity=nn.Tanh):
        super().__init__()
        """ RNNCell
            h = act(wh.h + bh + wx.x + bx)
            y = act(wy.h + by)
            author: girish d. hegde

        Args:
            input_size (int): input size.
            hidden_size (int): hidden size.
            bias (bool): add bias.
            nonlinearity (nn.Module): activation function.
        """
        self.i2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.h2o = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.hact = nonlinearity()
        self.yact = nonlinearity()

    def forward(self, input, hidden):
        """
        Args:
            input (torch.tensor): [b, l] - input data.
            hidden (torch.tensor): [b, h] - hidden state.

        Returns:
            tuple(torhc.tensor):
                torch.tensor: [b, h] - output.
                torch.tensor: [b, h] - updated hidden state.
        """
        h = self.i2h(input) + self.h2h(hidden)
        y = self.h2o(h)
        return self.yact(y), self.hact(h)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, ):
        super().__init__()
        """ RNN
            author: girish d. hegde

        Args:
            input_size (int): input size.
            hidden_size (int): hidden size.
            num_layers (int): number of rnn layers.
        """
        rnncells = [RNNCell(input_size, hidden_size)]  + \
            [RNNCell(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        self.rnncells = nn.ModuleList(rnncells)
        self.hidden_size = hidden_size

    def forward(self, x):
        """
        Args:
            x (torch.tensor): [timesteps, batchsize, input_size] - input data.

        Returns:
            tuple(torch.tensor):
                torch.tensor: [timestep, batchsize, hidden_size] - output.
                torch.tenosr: [num_layers, batchsize, hidden_size] - hidden state.
        """
        t, b, l = x.shape
        hidden = self.init_hidden(len(self.rnncells), b)
        out = []
        for timestep in range(t):
            output = x[timestep]
            for i, rnn_layer in enumerate(self.rnncells):
                output, hidden[i] = rnn_layer(output, hidden[i].clone())
            out.append(output)
        return torch.stack(out), hidden

    def init_hidden(self, num_layers, batch_size):
        return torch.zeros(num_layers, batch_size, self.hidden_size)


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
    out.sum().backward()
    print(f'{out.shape = }')
    print(f'{hdn.shape = }')