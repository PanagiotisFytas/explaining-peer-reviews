import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn


class BasicGRUClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=500, num_layers=1, pooling='last', bidirectional=False):
        super(BasicGRUClassifier, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                          batch_first=True, bidirectional=bidirectional)
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.pooling = pooling
        self.hx = None
        if self.bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        self.fc1 = nn.Linear(self.directions * hidden_size, 1)
        # self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        # self.relu2 = nn.ReLU()
        if self.pooling == 'attenetion_based':
            self.att = nn.Linear(self.directions * hidden_size, 1)

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
            out = self.hx.view(self.num_layers, self.directions, inp.shape[0], self.hidden_size)[-1]
            if self.bidirectional:
                # out = out[0] + out[1]
                out = torch.cat([out[0], out[1]], 1)
            else:
                out = out.squeeze()
        elif self.pooling == 'attention_based':
            # out, _output_lengths = rnn.pad_packed_sequence(out, batch_first=True, padding_value=-float("Inf"))
            out, _output_lengths = rnn.pad_packed_sequence(out, batch_first=True)
            out = out.transpose(1, 2)
            w_transpose_ht = self.att(out)
            a = nn.functional.softmax(w_transpose_ht, 2)  # batch dimension
        out = self.relu1(out)
        out = self.fc1(out)
        # out = self.relu2(out)
        # out = self.fc2(out)
        out = torch.sigmoid(out)
        return out


# this model is using 2 heads of attention
class AttentionClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_dimensions=[500]):
        super(AttentionClassifier, self).__init__()
        self.hidden_size = hidden_dimensions
        self.input_size = input_size
        self.fc_layers = nn.ModuleList([])
        layer_input = input_size*2
        for layer_out in hidden_dimensions:
            self.fc_layers.append(nn.Linear(layer_input, layer_out))
            layer_input = layer_out
        self.last_fc = nn.Linear(layer_input, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        # self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(0.5)
        self.att = nn.Linear(input_size, 1)
        self.att2 = nn.Linear(input_size, 1)

    def forward(self, inp, lengths):
        # seq, batch, embedding_dim = inp.shape
        w_t_h = self.att(inp)
        att = nn.functional.softmax(w_t_h, dim=1)

        out = torch.bmm(att.transpose(1, 2), inp).view(-1, self.input_size)  # out should be batch_size x emb_size

        w_t_h2 = self.att2(inp)
        att2 = nn.functional.softmax(w_t_h2, dim=1)

        out2 = torch.bmm(att2.transpose(1, 2), inp).view(-1, self.input_size)  # out should be batch_size x emb_size
        out = torch.cat([out, out2], dim=1)

        # out = self.relu(out)
        for layer in self.fc_layers:
            out = layer(out)
            out = self.relu(out)
            out = self.drop(out)

        out = self.last_fc(out)
        out = torch.sigmoid(out)
        return out
