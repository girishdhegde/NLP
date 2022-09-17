"""
Refs:
    https://colah.github.io/posts/2015-08-Understanding-LSTMs/
    https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html
"""


import torch
import torch.nn as nn


__author__ = "__Girish_Hegde__"


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        """ LSTMCell
                i = sigmoid(wii.x + bii + whi.h + bhi) - input gate.
                f = sigmoid(wif.x + bif + whf.h + bhf) - forget gate.
                o = sigmoid(wio.x + bio + who.h + bho) - output gate.
                g = tanh(wig.x + big + whg.h + bhg) - candidiates.
                c = f*c + i*g - updated cell state.
                h = o*tanh(c) - updated hidden/outout state.
            author: girish d. hegde

        Refs:
            https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html

        Args:
            input_size (int): input size.
            hidden_size (int): hidden size.
            bias (bool): add bias.
        """
        self.hidden_size = hidden_size
        self.i2f = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2f = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.i2i = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2i = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.i2o = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2o = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.i2g = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2g = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.fact = nn.Sigmoid()
        self.iact = nn.Sigmoid()
        self.oact = nn.Sigmoid()
        self.gact = nn.Tanh()
        self.yact = nn.Tanh()

    def forward(self, x, init_states=None):
        """
        Args:
            inp (torch.tensor): [b, l] - input data.
            init_states (tuple[torch.tensor]):
                torch.tensor: [b, h] - hidden state.
                torch.tensor: [b, h] - cell state.

        Returns:
            tuple(torhc.tensor):
                torch.tensor: [b, h] - output.
                torch.tensor: [b, h] - updated cell state.
        """
        ht, ct = self.init_hidden(x.shape[0], x.device) if init_states is None else init_states
        forget_gate = self.fact(self.i2f(x) + self.h2f(ht))
        input_gate = self.iact(self.i2i(x) + self.h2i(ht))
        output_gate = self.oact(self.i2o(x) + self.h2o(ht))
        candidiates = self.gact(self.i2g(x) + self.h2g(ht))
        next_ct = forget_gate*ct + input_gate*candidiates
        next_ht = output_gate*self.yact(next_ct)
        return next_ht, next_ct

    def init_hidden(self, batch_size, device='cpu'):
        h_t = torch.zeros(batch_size, self.hidden_size, device=device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=device)
        return (h_t, c_t)


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, ):
        super().__init__()
        """ LSTM
            author: girish d. hegde

        Refs:
            https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html

        Args:
            input_size (int): input size.
            hidden_size (int): hidden size.
            num_layers (int): number of lstm layers.
        """
        lstmcells = [LSTMCell(input_size, hidden_size)]  + \
            [LSTMCell(hidden_size, hidden_size) for _ in range(num_layers - 1)]
        self.lstmcells = nn.ModuleList(lstmcells)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x, init_states=None):
        """
        Args:
            x (torch.tensor): [seq_len, batchsize, input_size] - input data.
            init_states (tuple[torch.tensor]):
                list[torch.tensor]: [[batchsize, hidden_size] tensor for cell in self.lstmcells] - hidden states.
                list[torch.tensor]: [[batchsize, hidden_size] tensor for cell in self.lstmcells] - cell states.

        Returns:
            tuple(torch.tensor):
                torch.tensor: [seq_len, batchsize, hidden_size] - outputs.
                tuple(torch.tenosr): updated lstm states.
                    torch.tenosr: [num_layers, batchsize, hidden_size] - updated hidden states.
                    torch.tenosr: [num_layers, batchsize, hidden_size] - updated cell states.
        """
        seq_len, bs, l = x.shape
        hidden_states, cell_states = self.init_hidden(bs, x.device) if init_states is None else init_states
        out = []
        for t in range(seq_len):
            for i, lstm_layer in enumerate(self.lstmcells):
                x_t = x[t] if i == 0 else hidden_states[i - 1]
                hidden_states[i], cell_states[i] = lstm_layer(x_t, (hidden_states[i], cell_states[i]))
            out.append(hidden_states[-1])
        return torch.stack(out), (hidden_states, cell_states)

    def init_hidden(self, batch_size, device='cpu'):
        h_t, c_t = [], []
        for cell in self.lstmcells:
            h, c = cell.init_hidden(batch_size, device)
            h_t.append(h)
            c_t.append(c)
        return (h_t, c_t)


class WordLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2, dropout=0.0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = LSTM(input_size, hidden_size, num_layers)
        # self.lstm = nn.lstm(input_size, hidden_size, num_layers)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x, hidden=None):
        output, (h_t, c_t) = self.lstm(x, hidden)
        output = self.dropout(output)
        output = self.fc(output)
        return output, (h_t, c_t)

    def init_hidden(self, batch_size, device='cpu'):
        return self.lstm.init_hidden(batch_size, device)


if __name__ == '__main__':
    bs = 4
    timesteps = 5
    num_layers = 3
    input_size = 8
    hidden_size = 16

    # net = LSTM(input_size, hidden_size, num_layers)
    net = WordLSTM(input_size, hidden_size, num_layers)
    params = sum(p.numel() for p in net.parameters())
    # print(net)
    print(f'total parameters = {params} = {params/1e6}M')

    inp = torch.randn(timesteps, bs, input_size)
    init_states = net.init_hidden(bs, inp.device)
    out, (ht, ct) = net(inp, init_states)

    print(f'{out.shape = }')
    print(f'{len(ht) = }, {ht[0].shape = }')
    print(f'{len(ct) = }, {ct[0].shape = }')
    out.sum().backward()