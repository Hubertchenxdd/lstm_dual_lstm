import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTM(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, output_last = True):
        super(LSTM, self).__init__()

        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.il = nn.Linear(input_size + hidden_size, hidden_size)
        self.ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.Cl = nn.Linear(input_size + hidden_size, hidden_size)

        self.output_last = output_last

    def step(self, input, Hidden_State, Cell_State):
        combined = torch.cat((input, Hidden_State), 1)
        f = F.sigmoid(self.fl(combined))
        i = F.sigmoid(self.il(combined))
        o = F.sigmoid(self.ol(combined))
        C = F.tanh(self.Cl(combined))
        Cell_State = f * Cell_State + i * C
        Hidden_State = o * F.tanh(Cell_State)

        return Hidden_State, Cell_State

    def forward(self, inputs):
        batch_size = inputs.size(0)
        time_step = inputs.size(1)
        Hidden_State, Cell_State = self.initHidden(batch_size)

        if self.output_last:
            for i in range(time_step):
                Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:,i:i+1,:]), Hidden_State, Cell_State)
            return Hidden_State
        else:
            outputs = None
            for i in range(time_step):
                Hidden_State, Cell_State = self.step(torch.squeeze(inputs[:,i:i+1,:]), Hidden_State, Cell_State)
                if outputs is None:
                    outputs = Hidden_State.unsqueeze(1)
                else:
                    outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)
            return outputs

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State


class Dual_LSTM(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, output_last=True):
        super(Dual_LSTM, self).__init__()

        self.cell_size = cell_size
        self.hidden_size = hidden_size
        self.s_fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.s_il = nn.Linear(input_size + hidden_size, hidden_size)
        self.s_ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.s_Cl = nn.Linear(input_size + hidden_size, hidden_size)

        self.l_fl = nn.Linear(input_size + hidden_size, hidden_size)
        self.l_il = nn.Linear(input_size + hidden_size, hidden_size)
        self.l_ol = nn.Linear(input_size + hidden_size, hidden_size)
        self.l_Cl = nn.Linear(input_size + hidden_size, hidden_size)

        self.linear = nn.Linear(hidden_size + hidden_size, hidden_size)

        self.output_last = output_last

    def s_step(self, s_input, s_Hidden_State, s_Cell_State):
        s_combined = torch.cat((s_input, s_Hidden_State), 1)
        s_f = torch.sigmoid(self.s_fl(s_combined))
        s_i = torch.sigmoid(self.s_il(s_combined))
        s_o = torch.sigmoid(self.s_ol(s_combined))
        s_C = torch.tanh(self.s_Cl(s_combined))
        s_Cell_State = s_f * s_Cell_State + s_i * s_C
        s_Hidden_State = s_o * F.tanh(s_Cell_State)

        return s_Hidden_State, s_Cell_State

    def l_step(self, l_input, l_Hidden_State, l_Cell_State):
        l_combined = torch.cat((l_input, l_Hidden_State), 1)
        l_f = torch.sigmoid(self.l_fl(l_combined))
        l_i = torch.sigmoid(self.l_il(l_combined))
        l_o = torch.sigmoid(self.l_ol(l_combined))
        l_C = torch.tanh(self.l_Cl(l_combined))
        l_Cell_State = l_f * l_Cell_State + l_i * l_C
        l_Hidden_State = l_o * F.tanh(l_Cell_State)

        return l_Hidden_State, l_Cell_State

    def forward(self, s_inputs, l_inputs, dow, hour):
        batch_size = s_inputs.size(0)
        time_step = s_inputs.size(1)
        s_Hidden_State, s_Cell_State = self.initHidden(batch_size)
        l_Hidden_State, l_Cell_State = self.initHidden(batch_size)

        if self.output_last:
            for i in range(time_step):
                s_Hidden_State, s_Cell_State = self.s_step(torch.squeeze(s_inputs[:, i:i + 1, :]), s_Hidden_State,
                                                           s_Cell_State)
                l_Hidden_State, l_Cell_State = self.l_step(torch.squeeze(l_inputs[:, i:i + 1, :]), l_Hidden_State,
                                                           l_Cell_State)
            outputs = torch.tanh((self.linear(torch.cat((s_Hidden_State, l_Hidden_State), 1))))

            return outputs
        else:
            s_outputs = None
            l_outputs = None
            for i in range(time_step):
                s_Hidden_State, s_Cell_State = self.s_step(torch.squeeze(s_inputs[:, i:i + 1, :]), s_Hidden_State,
                                                           s_Cell_State)
                l_Hidden_State, l_Cell_State = self.l_step(torch.squeeze(l_inputs[:, i:i + 1, :]), l_Hidden_State,
                                                           l_Cell_State)

                if s_outputs is None:
                    s_outputs = s_Hidden_State.unsqueeze(1)
                else:
                    s_outputs = torch.cat((s_outputs, s_Hidden_State.unsqueeze(1)), 1)

                if l_outputs is None:
                    l_outputs = l_Hidden_State.unsqueeze(1)
                else:
                    l_outputs = torch.cat((l_outputs, l_Hidden_State.unsqueeze(1)), 1)
            outputs = torch.tanh(self.linear(torch.cat((s_outputs, l_outputs), 1)))

            return outputs

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda())
            return Hidden_State, Cell_State
        else:
            Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size))
            Cell_State = Variable(torch.zeros(batch_size, self.hidden_size))
            return Hidden_State, Cell_State