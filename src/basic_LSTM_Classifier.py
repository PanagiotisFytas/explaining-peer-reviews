import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from DataLoader import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report


GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size=768, hidden_size=500, num_layers=1):
        super(LSTMClassifier, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hx = None
        self.fc1 = nn.Linear(hidden_size, 1)
        # self.fc2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

    def forward(self, inp, lengths):
        # seq, batch, embedding_dim = inp.shape
        out = rnn.pack_padded_sequence(inp, lengths, enforce_sorted=False)  # no ONNX exportability
        # out.to(device)
        out, self.hx = self.gru(out)
        inp, _output_lengths = rnn.pad_packed_sequence(out)
        final_state = self.hx.view(self.num_layers, 1, inp.shape[1], self.hidden_size)[-1].squeeze()
        out = self.relu1(final_state)
        out = self.fc1(out)
        # out = self.relu2(out)
        # out = self.fc2(out)
        out = torch.sigmoid(out)
        return out


data_loader = DataLoader(device=device)
X = data_loader.read_embeddigns_from_file()
data = rnn.pad_sequence(X).to(device)
X_lengths = torch.tensor([reviews.shape[0] for reviews in X]).to(device)

labels = data_loader.read_labels().to(device)

shuffle = True
valid_size = 0.2

num_train = data.shape[1]
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

if shuffle:
    np.random.shuffle(indices)

train_idx, test_idx = indices[split:], indices[:split]

test_data = data[:, test_idx, :]
test_X_lengths = X_lengths[test_idx]
test_labels = labels[test_idx]

data = data[:, train_idx, :]
X_lengths = X_lengths[train_idx]
labels = labels[train_idx]


# print(X_lengths)
# X_prime = rnn.pad_sequence(X)
# print(X_prime.data.shape)
# seq_len, batch, input_size = X.data.shape
# X_2 = rnn.pack_sequence(X, enforce_sorted=False)  # no ONNX exportability

# input_size = 768
# N = 427

_seq_len, N, input_size = data.shape
_, test_N, _ = test_data.shape
print(data.shape, test_data.shape)

model = LSTMClassifier(input_size=input_size, hidden_size=500, num_layers=2)
model.to(device)
# print(labels)
# print(len(labels))
#
# print(model(X[:10]))

epochs = 100
batch_size = 16
lr = 0.0001

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.BCELoss()

for epoch in range(1, epochs+1):
    permutation = torch.randperm(N)
    model.train()
    for i in range(0, N, batch_size):
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        batch_y = labels[indices]
        batch_x = data[:, indices, :]
        batch_x.to(device)
        batch_y.to(device)


        batch_lengths = X_lengths[indices]
        batch_lengths.to(device)

        preds = model(batch_x, batch_lengths).squeeze(1)
        loss = loss_fn(preds, batch_y)
        train_loss = loss.item()
        loss.backward()
        model.hx = model.hx.detach()
        optimizer.step()

    model.eval()

    predictions = model(data, X_lengths)
    preds = predictions.view(-1) >= 0.5
    targets = labels >= 0.5

    accuracy = (preds == targets).sum() * (1 / N)
    print('-----EPOCH ' + str(epoch) + '-----')
    print('Accuracy on train set: ', accuracy)

    predictions = model(test_data, test_X_lengths)
    preds = (predictions.view(-1) >= 0.5).to(device='cpu', dtype=torch.int)
    targets = (test_labels >= 0.5).to(device='cpu', dtype=torch.int)

    # print(preds.shape)
    # print(targets.shape)
    accuracy = (preds == targets).sum() * (1/test_N)
    print('Accuracy on test set: ', accuracy)
    print('Confusion on test set: ', confusion_matrix(targets.numpy(), preds.numpy()))
    print('Precision on test set: ', precision_score(targets.numpy(), preds.numpy()))
    print('Recall on test set: ', recall_score(targets, preds))
    print('F1 on test set: ', f1_score(targets, preds))
    print('Report:\n', classification_report(targets, preds))
    print('-----------------')
    # predictions = model(test_data).squeeze(1)
    # print('RMSE on test set: ', rmse(predictions, test_labels))


