import torch
import torch.nn as nn

class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size = 64, num_layers = 3, out_size = None):
        super(LSTMNet, self).__init__()
        
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        if out_size is None:
            self.out_size = input_size
        else:
            self.out_size = out_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first = True)

        self.linear1 = nn.Linear(hidden_size, 2 * hidden_size)
        self.linear2 = nn.Linear(2 * hidden_size, self.out_size)

    def forward(self, x, last = None):
        # x.shape = batch_size, sequence_length, embedding size
        if last is None:
            x, (hn, cn) = self.lstm(x)
        else:
            x, (hn, cn) = self.lstm(x, last)
        x = self.linear1(x)
        x = nn.functional.relu(x)
        x = self.linear2(x)
        return x, (hn, cn)


class Classifier(nn.Module):
    def __init__(self, lstm, out_size):
        super(Classifier, self).__init__()
        
        self.lstm = lstm
        self.num_layers = lstm.num_layers
        self.input_size = lstm.input_size
        self.hidden_size = lstm.hidden_size
        self.acti = nn.ReLU()
        self.linear = nn.Linear(lstm.input_size, out_size)

    def forward(self, x, last = None):
        # x.shape = batch_size, sequence_length, embedding size
        x, (hn, cn) = self.lstm(x, last)
        x = self.acti(x)
        x = self.linear(x)
        return x, (hn, cn)