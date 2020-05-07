import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class BasicGRUClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=500, num_layers=1, pooling='last'):
        super(BasicGRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                          batch_first=True)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pooling = pooling
        self.hx = None
        self.fc1 = nn.Linear(hidden_size, 1)
        # self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, inp, lengths):
        # seq, batch, embedding_dim = inp.shape
        out = rnn.pack_padded_sequence(inp, lengths, enforce_sorted=False, batch_first=True)  # no ONNX exportability
        # out.to(device)
        out, self.hx = self.gru(out)
        if self.pooling == 'max':
            out, _output_lengths = rnn.pad_packed_sequence(out, batch_first=True, padding_value=-float("Inf"))
            out = out.transpose(1, 2)
            out = nn.functional.max_pool1d(out, kernel_size=out.shape[2]).squeeze()
        elif self.pooling == 'avg':
            out, _output_lengths = rnn.pad_packed_sequence(out, batch_first=True)
            out = out.transpose(1, 2)
            out = nn.functional.avg_pool1d(out, kernel_size=out.shape[2]).squeeze()
        elif self.pooling == 'last':
            out = self.hx.view(self.num_layers, 1, inp.shape[0], self.hidden_size)[-1].squeeze()
        out = self.relu1(out)
        out = self.fc1(out)
        # out = self.relu2(out)
        # out = self.fc2(out)
        out = torch.sigmoid(out)
        return out
