import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from pytorch_revgrad import RevGrad


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
    def __init__(self, dropout=0.2, input_size=768, hidden_dimensions=[500]):
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
        self.activation= nn.ReLU()
        self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(dropout)
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
            out = self.activation(out)
            out = self.drop(out)

        out = self.last_fc(out)
        out = torch.sigmoid(out)
        return out


class MultiheadClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_dimensions=[500], heads=1):
        super(MultiheadClassifier, self).__init__()
        self.hidden_size = hidden_dimensions
        self.input_size = input_size
        self.heads = heads
        self.fc_layers = nn.ModuleList([])
        layer_input = input_size*self.heads
        for layer_out in hidden_dimensions:
            self.fc_layers.append(nn.Linear(layer_input, layer_out))
            layer_input = layer_out
        self.last_fc = nn.Linear(layer_input, 1)
        self.sigmoid = nn.Sigmoid()
        self.activation= nn.Tanh()
        # self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.att_layers = nn.ModuleList([])
        for _ in range(heads):
            self.att_layers.append(nn.Linear(input_size, 1))

    def forward(self, inp, lengths):
        # seq, batch, embedding_dim = inp.shape
        outs = []
        for att in self.att_layers:
            w_t_h = att(inp)
            att = nn.functional.softmax(w_t_h, dim=1)
            out = torch.bmm(att.transpose(1, 2), inp).view(-1, self.input_size)  # out should be batch_size x emb_size
            outs.append(out)

        out = torch.cat(outs, dim=1)

        # out = self.relu(out)
        for layer in self.fc_layers:
            out = layer(out)
            out = self.activation(out)
            out = self.drop(out)

        out = self.last_fc(out)
        out = torch.sigmoid(out)
        return out


class ScoreClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_dimensions=[500]):
        super(ScoreClassifier, self).__init__()
        self.hidden_size = hidden_dimensions
        self.input_size = input_size
        self.fc_layers = nn.ModuleList([])
        layer_input = input_size
        for layer_out in hidden_dimensions:
            self.fc_layers.append(nn.Linear(layer_input, layer_out))
            layer_input = layer_out
        self.last_fc = nn.Linear(layer_input, 1)
        self.sigmoid = nn.Sigmoid()
        self.activation= nn.Tanh()
        # self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(0.1)

    def forward(self, inp, lengths):
        out = inp

        # out = self.relu(out)
        for layer in self.fc_layers:
            out = layer(out)
            out = self.activation(out)
            out = self.drop(out)

        out = self.last_fc(out)
        out = torch.sigmoid(out)
        return out


class AbstractClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_dimensions=[500]):
        super(AbstractClassifier, self).__init__()
        self.hidden_size = hidden_dimensions
        self.input_size = input_size
        self.fc_layers = nn.ModuleList([])
        layer_input = input_size
        for layer_out in hidden_dimensions:
            self.fc_layers.append(nn.Linear(layer_input, layer_out))
            layer_input = layer_out
        self.last_fc = nn.Linear(layer_input, 1)
        self.sigmoid = nn.Sigmoid()
        self.activation= nn.Tanh()
        # self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(0.1)

    def forward(self, inp, lengths):
        out = inp

        # out = self.relu(out)
        for layer in self.fc_layers:
            out = layer(out)
            out = self.activation(out)
            out = self.drop(out)

        out = self.last_fc(out)
        out = torch.sigmoid(out)
        return out


class LSTMAttentionClassifier(nn.Module):
    def __init__(self, device, input_size=768, lstm_hidden_size=500, num_layers=1, bidirectional=False, hidden_dimensions=[500],
                 cell_type='GRU', causal_layer=None, causal_hidden_dimensions=[30, 20], att_dim=30, dropout1=0.2,
                 dropout2=0.2, activation='ReLU', adversarial_out=None, task='classification'):
        super(LSTMAttentionClassifier, self).__init__()
        self.task = task
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        self.adversarial_out = adversarial_out
        self.device = device
        self.cell_type = cell_type
        if cell_type == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=lstm_hidden_size, num_layers=num_layers, dropout=0.5,
                              batch_first=True, bidirectional=bidirectional)
            self.hx = None
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=lstm_hidden_size, num_layers=num_layers, dropout=0.5,
                                batch_first=True, bidirectional=bidirectional)
            self.hx = None
            self.cx = None
        else:
            raise Exception('Invalid RNN type')
        self.bidirectional = bidirectional
        self.lstm_hidden_size = lstm_hidden_size
        self.num_layers = num_layers
        self.hidden_dimensions = hidden_dimensions
        if self.bidirectional:
            self.directions = 2
        else:
            self.directions = 1
        if causal_layer and causal_layer == 'residual':
            layer_input = lstm_hidden_size + 1
            # layer_input = lstm_hidden_size + 768 # + causal_hidden_dimensions[-1]
        else:
            layer_input = lstm_hidden_size * self.directions
        self.fc_layers = nn.ModuleList([])
        for layer_out in hidden_dimensions:
            self.fc_layers.append(nn.Linear(layer_input, layer_out))
            layer_input = layer_out
        self.last_fc = nn.Linear(layer_input, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(self.dropout1)
        if activation == 'ReLU':
            self.activation = nn.ReLU()
        else:
            self.activation = nn.Tanh()

        self.att1 = nn.Linear(self.directions * lstm_hidden_size, att_dim, bias=False)
        self.att2 = nn.Linear(att_dim, 1, bias=False)

        self.causal_layer = causal_layer
        if causal_layer == 'adversarial':
            self.rev = RevGrad()
            self.drop2 = nn.Dropout(self.dropout2)
            layer_input = lstm_hidden_size * self.directions
            self.causal_layers = nn.ModuleList([])
            for layer_out in causal_hidden_dimensions:
                self.causal_layers.append(nn.Linear(layer_input, layer_out))
                layer_input = layer_out
            if not adversarial_out:
                self.causal_last_fc = nn.Linear(layer_input, 10) # regression as multiclass classification
                self.classes = torch.arange(1, 11).view(-1, 1).to(self.device, dtype=torch.float) # classes has shape 10, 1
                self.softmax = nn.Softmax()
            else:
                # adversarial out is a tuple of (number_of_confounders, ids of confounders with sigmoid)
                self.causal_last_fc = nn.Linear(layer_input, adversarial_out[0])

        elif causal_layer == 'residual':
            self.drop2 = nn.Dropout(self.dropout2)
            if not adversarial_out:
                layer_input = input_size
            else:
                layer_input = adversarial_out[0]
            self.causal_layers = nn.ModuleList([])
            for layer_out in causal_hidden_dimensions:
                self.causal_layers.append(nn.Linear(layer_input, layer_out))
                layer_input = layer_out
            self.causal_last_fc = nn.Linear(layer_input, 1)

    def create_mask(self, lengths):
        max_len = lengths.max()
        mask = (torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1))
        return mask.unsqueeze(2).to(self.device, dtype=torch.int)

    def forward(self, inp, lengths, abstract=None):
        # forward through the rnn and get the output of the rnn and the attention weights
        if self.causal_layer == 'residual':
            confounding_out, out_vector = self.residual_mlp_forward(abstract)

        attention, rnn_out = self.rnn_att_forward(inp, lengths)
        # print(attention.shape)
        # pool the output of the rnn using attention
        rnn_out = torch.bmm(attention.transpose(1, 2), rnn_out).view(-1, self.lstm_hidden_size * self.directions)  # out should be batch_size x lstm_hidden_size
        # print(out.shape)

        if self.causal_layer == 'residual':
            # print(rnn_out.shape, confounding_out.shape)
            out = torch.cat([rnn_out, confounding_out], dim=1)
            # out = torch.cat([rnn_out, out_vector], dim=1)
        else:
            out = rnn_out
        
        for layer in self.fc_layers:
            out = self.drop(out)
            out = layer(out)
            out = self.activation(out)
        out = self.drop(out)

        out = self.last_fc(out)
        if self.task == 'classification':
            out = self.sigmoid(out)

        if not self.causal_layer:
            return out
        elif self.causal_layer == 'adversarial':
            confounding_out = self.causal_mlp_forward(rnn_out)
            return out, confounding_out
        elif self.causal_layer == 'residual':
            return out, confounding_out
        
    def rnn_att_forward(self, inp, lengths):
        # seq, batch, embedding_dim = inp.shape
        out = rnn.pack_padded_sequence(inp, lengths, enforce_sorted=False, batch_first=True)  # no ONNX exportability
        # out.to(device)
        if self.cell_type == 'GRU':
            out, self.hx = self.rnn(out)  # no hx input since no BPTT
        elif self.cell_type == 'LSTM':
            out, (self.hx, self.cx) = self.rnn(out)
    
        out, output_lengths = rnn.pad_packed_sequence(out, batch_first=True)
        
        mask = self.create_mask(output_lengths)

        attention = self.att1(out)
        attention = self.activation(attention)
        attention = self.att2(attention)
        # attention = self.activation(attention)
        # print('Mask: ', mask.shape)
        # print('Att layer output: ', attention.shape)
        attention = attention.masked_fill(mask==0, -1e10)  # values of the mask (equal to 0) will become -10^10 so in softmax they are zero
        # print(attention.shape)
        # calculate attention weights
        attention = nn.functional.softmax(attention, dim=1)
        
        return attention, out

    def causal_mlp_forward(self, rnn_out):
        out = self.rev(rnn_out)
        for layer in self.causal_layers:
            out = self.drop2(out)
            out = layer(out)
            out = self.activation(out)
        out = self.drop2(out)
        out = self.causal_last_fc(out)
        # out = self.sigmoid(out)
        if not self.adversarial_out:
            out = self.softmax(out)
            out = out.matmul(self.classes) # get weights average of age for regression
        else:
            for idx in self.adversarial_out[1]:
                out[:, idx] = torch.nn.functional.sigmoid(out[:, idx])
        return out

    def residual_mlp_forward(self, abstract):
        out = abstract
        for layer in self.causal_layers:
            out = self.drop2(out)
            out = layer(out)
            out = self.activation(out)
        out_vector = out
        # out = self.drop2(out)
        out = self.causal_last_fc(out)
        if self.task == 'classification':
            out = self.sigmoid(out)
        return out, abstract

class BERTClassifier(nn.Module):
    def __init__(self, device, input_size=768, hidden_dimensions=[16],
                 causal_layer=None, causal_hidden_dimensions=[10, 5], BERT_hidden_dimensions=[100, 30], dropout=0.2,
                 activation='Tanh', activation2='Tanh', dropout2=0.2):
        super(BERTClassifier, self).__init__()
        self.hidden_size = hidden_dimensions
        self.input_size = input_size
        # bert mlp
        self.bert_fc_layers = nn.ModuleList([])
        layer_input = input_size
        for layer_out in BERT_hidden_dimensions:
            self.bert_fc_layers.append(nn.Linear(layer_input, layer_out))
            layer_input = layer_out
        
        # last mlp
        self.fc_layers = nn.ModuleList([])
        if causal_layer == 'residual':
            layer_input = BERT_hidden_dimensions[-1] + 1
        else:
            layer_input = BERT_hidden_dimensions[-1]
        for layer_out in hidden_dimensions:
            self.fc_layers.append(nn.Linear(layer_input, layer_out))
            layer_input = layer_out
        self.last_fc = nn.Linear(layer_input, 1)
        self.sigmoid = nn.Sigmoid()
        if activation == 'Tanh':
            self.activation= nn.Tanh()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.causal_layer = causal_layer
        self.drop2 = nn.Dropout(dropout2)
        # residual mlp
        if causal_layer == 'residual':
            if activation2 == 'Tanh':
                self.activation2 = nn.Tanh()
            elif activation2 == 'ReLU':
                self.activation2 = nn.ReLU()
        
            layer_input = input_size
            self.causal_layers = nn.ModuleList([])
            if causal_hidden_dimensions:
                for layer_out in causal_hidden_dimensions:
                    self.causal_layers.append(nn.Linear(layer_input, layer_out))
                    layer_input = layer_out
            self.causal_last_fc = nn.Linear(layer_input, 1)


    def forward(self, inp, _lengths, abstract=None):

        if self.causal_layer == 'residual':
            confounding_out, _ = self.residual_mlp_forward(abstract)


        out = self.bert_mlp_forward(inp)
        if self.causal_layer == 'residual':
            # print(rnn_out.shape, confounding_out.shape)
            out = torch.cat([out, confounding_out], dim=1)
            # out = torch.cat([rnn_out, out_vector], dim=1)

        # out = self.relu(out)
        for layer in self.fc_layers:
            out = self.drop2(out)
            out = layer(out)
            out = self.activation(out)
        out = self.drop2(out)
        out = self.last_fc(out)
        out = torch.sigmoid(out)
        
        if self.causal_layer == 'residual':
            return out, confounding_out
        else:
            return out

    def residual_mlp_forward(self, abstract):
        out = abstract
        for layer in self.causal_layers:
            out = self.drop2(out)
            out = layer(out)
            out = self.activation2(out)
        # out_vector = out
        out = self.drop2(out)
        out = self.causal_last_fc(out)
        out = self.sigmoid(out)
        return out, abstract

    def bert_mlp_forward(self, inp):
        out = inp
        for layer in self.bert_fc_layers:
            out = self.drop(out)
            out = layer(out)
            out = self.activation(out)
        return out
        

class BoWClassifier(nn.Module):
    def __init__(self, device, input_size=6290, embedding_size=768, hidden_dimensions=[16],
                 causal_layer=None, causal_hidden_dimensions=[10, 5], bow_hidden_dimensions=[30], dropout=0.2,
                 activation='Tanh', activation2='Tanh', dropout2=0.2):
        super(BoWClassifier, self).__init__()
        self.hidden_size = hidden_dimensions
        self.input_size = input_size
        # first mlp
        self.bow_fc_layers = nn.ModuleList([])
        layer_input = input_size
        for layer_out in bow_hidden_dimensions:
            self.bow_fc_layers.append(nn.Linear(layer_input, layer_out))
            layer_input = layer_out
        
        # last mlp
        self.fc_layers = nn.ModuleList([])
        if causal_layer == 'residual':
            layer_input = bow_hidden_dimensions[-1] + 1
        else:
            layer_input = bow_hidden_dimensions[-1]
        for layer_out in hidden_dimensions:
            self.fc_layers.append(nn.Linear(layer_input, layer_out))
            layer_input = layer_out
        self.last_fc = nn.Linear(layer_input, 1)
        self.sigmoid = nn.Sigmoid()
        if activation == 'Tanh':
            self.activation= nn.Tanh()
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.causal_layer = causal_layer
        # residual mlp
        if causal_layer == 'residual':
            if activation2 == 'Tanh':
                self.activation2 = nn.Tanh()
            elif activation2 == 'ReLU':
                self.activation2 = nn.ReLU()
        
            self.drop2 = nn.Dropout(dropout2)
            layer_input = embedding_size
            self.causal_layers = nn.ModuleList([])
            if causal_hidden_dimensions:
                for layer_out in causal_hidden_dimensions:
                    self.causal_layers.append(nn.Linear(layer_input, layer_out))
                    layer_input = layer_out
            self.causal_last_fc = nn.Linear(layer_input, 1)


    def forward(self, inp, _lengths, abstract=None):

        if self.causal_layer == 'residual':
            confounding_out, _ = self.residual_mlp_forward(abstract)


        out = self.bow_mlp_forward(inp)
        if self.causal_layer == 'residual':
            # print(rnn_out.shape, confounding_out.shape)
            out = torch.cat([out, confounding_out], dim=1)
            # out = torch.cat([rnn_out, out_vector], dim=1)

        # out = self.relu(out)
        for layer in self.fc_layers:
            out = self.drop(out)
            out = layer(out)
            out = self.activation(out)

        out = self.last_fc(out)
        out = torch.sigmoid(out)
        
        if self.causal_layer == 'residual':
            return out, confounding_out
        else:
            return out

    def residual_mlp_forward(self, abstract):
        out = abstract
        for layer in self.causal_layers:
            out = self.drop2(out)
            out = layer(out)
            out = self.activation2(out)
        # out_vector = out
        out = self.drop2(out)
        out = self.causal_last_fc(out)
        out = self.sigmoid(out)
        return out, abstract

    def bow_mlp_forward(self, inp):
        out = inp
        for layer in self.bow_fc_layers:
            out = self.drop(out)
            out = layer(out)
            out = self.activation(out)
        return out
        
