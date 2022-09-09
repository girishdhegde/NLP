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

        Refs:
            https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks
            https://pytorch.org/docs/stable/generated/torch.nn.RNN.html

        Args:
            input_size (int): input size.
            hidden_size (int): hidden size.
            num_layers (int): number of rnn layers.
        """
        rnncells = [RNNCell(input_size, hidden_size)]  + \
            [RNNCell(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        self.rnncells = nn.ModuleList(rnncells)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, hidden):
        """
        Args:
            x (torch.tensor): [timesteps, batchsize, input_size] - input data.
            hidden (torch.tensor): [num_layers, batchsize, hidden_size] - input hidden state.

        Returns:
            tuple(torch.tensor):
                torch.tensor: [timestep, batchsize, hidden_size] - output.
                torch.tenosr: [num_layers, batchsize, hidden_size] - hidden state.
        """
        t, b, l = x.shape
        out = []
        for timestep in range(t):
            output = x[timestep]
            for i, rnn_layer in enumerate(self.rnncells):
                output, hidden[i] = rnn_layer(output, hidden[i].clone())
            out.append(output)
        return torch.stack(out), hidden

    def init_hidden(self, batch_size, device='cpu'):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.1):
        super().__init__()
        self.rnn = RNN(input_size, hidden_size, num_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, input_size)


    def forward(self, x, hidden):
        output, hdn = self.rnn(x, hidden)
        output = self.dropout(output)
        output = self.fc(output)
        return output, hdn


if __name__ == '__main__':
    bs = 4
    timesteps = 5
    num_layers = 3
    input_size = 26
    hidden_size = 64

    # net = RNN(input_size, hidden_size, num_layers)
    net = CharRNN(input_size, hidden_size, num_layers)
    params = sum(p.numel() for p in net.parameters())
    print(net)
    print(f'total parameters = {params} = {params/1e6}M')

    inp = torch.randn(timesteps, bs, input_size)
    hdn = net.rnn.init_hidden(inp.shape[1], inp.device)
    out, hdn = net(inp, hdn)
    out.sum().backward()
    print(f'{out.shape = }')
    print(f'{hdn.shape = }')