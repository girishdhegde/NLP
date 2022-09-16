import torch
import torch.nn as nn
import  torch.nn.functional as F


class Gate(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='sigmoid', *args, **kwargs):
        super().__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.nonlinearity = getattr(F, nonlinearity)

    def forward(self, input, hidden):
        # h = act(wh.h + bh + wx.x + bx)
        h = self.i2h(input) + self.h2h(hidden)
        return self.nonlinearity(h)


class ForgetGate(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.gate = Gate(input_size, hidden_size, bias=bias, nonlinearity='sigmoid')

    def forward(self, input, hidden):
        return self.gate(input, hidden)


class InputGate(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.mask = Gate(input_size, hidden_size, bias=bias, nonlinearity='sigmoid')
        self.new_state = Gate(input_size, hidden_size, bias=bias, nonlinearity='tanh')

    def forward(self, input, hidden):
        it = self.mask(input, hidden)
        ct = self.new_state(input, hidden)
        return it*ct


class OutputGate(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super().__init__()
        self.mask = Gate(input_size, hidden_size, bias=bias, nonlinearity='sigmoid')

    def forward(self, input, hidden, cell_state):
        ot = self.mask(input, hidden)
        return ot*torch.tanh(cell_state)


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True, ):
        super().__init__()

        self.forget_gate = ForgetGate(input_size, hidden_size, bias=bias)
        self.input_gate = InputGate(input_size, hidden_size, bias=bias)
        self.output_gate = OutputGate(input_size, hidden_size, bias=bias)

    def forward(self, input, hidden, cell_state):
        # Forget gate
        ft = self.forget_gate(input, hidden)
        # Input gate
        it = self.input_gate(input, hidden)
        # Update cell state
        ct = ft*cell_state + it
        # Output gate
        ht = self.output_gate(input, hidden, ct)
        # return hidden, cell_state
        return ht, ct


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True):
        super().__init__()

        lstms = [LSTMCell(input_size, hidden_size, bias)]  + \
            [LSTMCell(hidden_size, hidden_size, bias) for _ in range(num_layers - 1)]
        self.lstms = nn.ModuleList(lstms)
        self.hidden_size = hidden_size

    def forward(self, x):
        '''
        Args:
            x: input tensor of size [Sequence length, batch size, input size]
        Returns:
            out: output tensor of size [Sequence length, batch size, hidden size]
            cell_state: Final cell state of size [batch size, hidden size]
        '''
        out = torch.zeros(x.shape[0], x.shape[1], self.hidden_size)
        hidden_states = torch.zeros(x.shape[0] + 1, x.shape[1], self.hidden_size)
        cell_states = torch.zeros(x.shape[0] + 1, x.shape[1], self.hidden_size)

        for timestep in range(x.shape[0]):
            hidden = x[timestep]
            
            for lstm_layer in self.lstms:
                hidden, cell_state = lstm_layer(
                    hidden, 
                    hidden_states[timestep], 
                    cell_states[timestep]
                )

            out[timestep] = hidden
            hidden_states[timestep + 1] = hidden
            cell_states[timestep + 1] = cell_state

        return out, hidden_states[1:], cell_states[1:]


if __name__ == '__main__':
    bs = 4
    timesteps = 3
    num_layers = 2
    input_size = 10
    hidden_size = 20

    net = LSTM(input_size, hidden_size, num_layers)
    print(net)

    inp = torch.randn(timesteps, bs, input_size)
    out, hts, cts = net(inp)
    print(out.shape, hts.shape, cts.shape)
