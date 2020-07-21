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
        # self.activation= nn.Tanh()
        self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(0.2)
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

# class CausalBERT(nn.Module):
#     """Fine tune Bert with for causal inference
#     """
#     DATA_ROOT = pathlib.Path(os.environ['DATA'])
#     SCIBERT_PATH = str(DATA_ROOT / 'scibert_scivocab_uncased')
#     def __init__(self, freeze_bert=False):
#         """
#         @param    bert: a BertModel object
#         @param    classifier: a torch.nn.Module classifier
#         @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
#         """
#         super(BertClassifier, self).__init__()
#         # Specify hidden size of BERT, hidden size of our classifier, and number of labels
#         D_in, H, D_out = 768, 200, 2

#         # Instantiate BERT model
#         self.bert = BertModel.from_pretrained(self.SCIBERT_PATH)

#         # Instantiate an one-layer feed-forward classifier
#         self.propensity = nn.Sequential(
#             nn.Linear(D_in, 1)
#         )

#         self.expectations == nn.ModuleList([
#             nn.Sequential( # for treatment = 0
#                 nn.Linear(D_in, H),
#                 nn.ELU(),
#                 nn.Linear(H, 1)
#             ),
#             nn.Sequential( # for treatment = 1
#                 nn.Linear(D_in, H),
#                 nn.ELU(),
#                 nn.Linear(H, 1)
#             )

#         ])

#         # Freeze the BERT model
#         if freeze_bert:
#             for param in self.bert.parameters():
#                 param.requires_grad = False
        
#     def forward(self, input_ids, attention_mask):
#         """
#         Feed input to BERT and the classifier to compute logits.
#         @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
#                       max_length)
#         @param    attention_mask (torch.Tensor): a tensor that hold attention mask
#                       information with shape (batch_size, max_length)
#         @return   logits (torch.Tensor): an output tensor with shape (batch_size,
#                       num_labels)
#         """
#         # Feed input to BERT
#         outputs = self.bert(input_ids=input_ids,
#                             attention_mask=attention_mask)
        
#         # Extract the last hidden state of the token `[CLS]` for classification task
#         last_hidden_state_cls = outputs[0][:, 0, :]

#         # Feed input to classifier to compute logits for propensity scores
#         logits = self.classifier(last_hidden_state_cls)
#         log_probs = torch.nn.logsoftmax(logits)
#         propensity = torch.nn.softmax(logits) # P(T=1)
#         # the loss will need logsoftmax

#         q_t_0 = self.expectations[0](last_hidden_state_cls) # Q(0,lambda;gamma) = E[Y|T=0,Abstract]
#         q_t_1 = self.expectations[1](last_hidden_state_cls) # Q(1,lambda;gamma) = E[Y|T=1,Abstract]

#         return logits

class LSTMAttentionClassifier(nn.Module):
    def __init__(self, device, input_size=768, lstm_hidden_size=500, num_layers=1, bidirectional=False, hidden_dimensions=[500],
                 cell_type='GRU', causal_layer=None, causal_hidden_dimensions=[30, 20], att_dim=30):
        super(LSTMAttentionClassifier, self).__init__()
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
        else:
            layer_input = lstm_hidden_size * self.directions
        self.fc_layers = nn.ModuleList([])
        for layer_out in hidden_dimensions:
            self.fc_layers.append(nn.Linear(layer_input, layer_out))
            layer_input = layer_out
        self.last_fc = nn.Linear(layer_input, 1)
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(0.2)
        self.activation = nn.ReLU()
        
        self.att1 = nn.Linear(self.directions * lstm_hidden_size, att_dim, bias=False)
        self.att2 = nn.Linear(att_dim, 1, bias=False)

        self.causal_layer = causal_layer
        if causal_layer == 'adversarial':
            self.drop2 = nn.Dropout(0.2)
            layer_input = lstm_hidden_size * self.directions
            self.causal_layers = nn.ModuleList([])
            for layer_out in causal_hidden_dimensions:
                self.causal_layers.append(nn.Linear(layer_input, layer_out))
                layer_input = layer_out
            self.causal_last_fc = nn.Linear(layer_input, 10) # regression as multiclass classification
            self.classes = torch.arange(1, 11).view(-1, 1).to(self.device, dtype=torch.float) # classes has shape 10, 1
            self.softmax = nn.Softmax()
        elif causal_layer == 'residual':
            self.drop2 = nn.Dropout(0.2)
            layer_input = input_size
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
            confounding_out = self.residual_mlp_forward(abstract)

        attention, rnn_out = self.rnn_att_forward(inp, lengths)
        # print(attention.shape)
        # pool the output of the rnn using attention
        rnn_out = torch.bmm(attention.transpose(1, 2), rnn_out).view(-1, self.lstm_hidden_size * self.directions)  # out should be batch_size x lstm_hidden_size
        # print(out.shape)

        if self.causal_layer == 'residual':
            # print(rnn_out.shape, confounding_out.shape)
            out = torch.cat([rnn_out, confounding_out], dim=1)
        else:
            out = rnn_out
        
        for layer in self.fc_layers:
            out = self.drop(out)
            out = layer(out)
            out = self.activation(out)
        out = self.drop(out)

        out = self.last_fc(out)
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
        attention = self.activation(out)
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
        out = rnn_out
        for layer in self.causal_layers:
            out = self.drop2(out)
            out = layer(out)
            out = self.activation(out)
        out = self.drop2(out)
        out = self.causal_last_fc(out)
        # out = self.sigmoid(out)
        out = self.softmax(out)
        out = out.matmul(self.classes) # get weights average of age for regression
        return out

    def residual_mlp_forward(self, abstract):
        out = abstract
        for layer in self.causal_layers:
            out = self.drop2(out)
            out = layer(out)
            out = self.activation(out)
        out = self.drop2(out)
        out = self.causal_last_fc(out)
        out = self.sigmoid(out)
        return out