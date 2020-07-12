import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from DataLoader import LSTMEmbeddingLoader
import numpy as np
from helper_functions import training_loop, cross_validation_metrics
from models import LSTMAttentionClassifier
import pathlib
import os
import matplotlib.pyplot as plt


device_idx = input("GPU: ")
GPU = True
if GPU:
    device = torch.device("cuda:" + device_idx if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

cross_validation = False
# cross_validation = True


data_loader = LSTMEmbeddingLoader(device=device,
                                  lemmatise=True, 
                                  lowercase=True, 
                                  stopword_removal=False, 
                                  punctuation_removal=True,
                                  final_decision='only',
                                  pretrained_weights='scibert_scivocab_uncased',
                                 )

try:
    embeddings_input = data_loader.read_embeddigns_from_file()
except FileNotFoundError:
    # create file with embeddings if it does not exist
    embeddings_input = data_loader.get_embeddings_from_reviews()
    data_loader.write_embeddings_to_file()


number_of_tokens = torch.tensor([review.shape[0] for review in embeddings_input]).to(device)
embeddings_input = rnn.pad_sequence(embeddings_input, batch_first=True).to(device)  # pad the reviews to form a tensor
print(embeddings_input.shape)
labels = data_loader.read_labels().to(device)
_, _, embedding_dimension = embeddings_input.shape

epochs = 100 # 110 # 500
batch_size = 100 # 30
lr = 0.0001
hidden_dimensions = [128, 64] # [128, 64] # [1500, 700, 300]
lstm_hidden_dimension = 300 # 500
num_layers = 1  # Layers in the RN. Having more than 1 layer probably makes interpretability worst by combining more tokens into hiddent embs
bidirectional = False
cell_type = 'GRU'

if cross_validation:
    network = LSTMAttentionClassifier
    network_params = {
        'device': device, 
        'input_size': embedding_dimension,
        'lstm_hidden_size': lstm_hidden_dimension,
        'num_layers': num_layers,
        'bidirectional': bidirectional,
        'hidden_dimensions': hidden_dimensions,
        'cell_type': cell_type
    }
    optimizer = torch.optim.Adam
    lr = lr
    loss_fn = nn.BCELoss
    data = [embeddings_input, number_of_tokens, labels]
    cross_validation_metrics(network, network_params, optimizer, loss_fn, lr,
                             epochs, batch_size, device, data, k=5, shuffle=True)
else:
    # hold-one-out split
    model = LSTMAttentionClassifier(device=device, 
                                    input_size=embedding_dimension,
                                    lstm_hidden_size=lstm_hidden_dimension,
                                    num_layers=num_layers,
                                    bidirectional=bidirectional,
                                    hidden_dimensions=hidden_dimensions,
                                    cell_type=cell_type
                                   )
    shuffle = False
    valid_size = 0.1
    print(embeddings_input.shape)

    num_train = embeddings_input.shape[0]
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]

    test_embeddings_input = embeddings_input[test_idx, :, :]
    test_number_of_tokens = number_of_tokens[test_idx]
    test_labels = labels[test_idx]

    embeddings_input = embeddings_input[train_idx, :, :]
    number_of_tokens = number_of_tokens[train_idx]
    labels = labels[train_idx]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    data = [embeddings_input, number_of_tokens, labels]
    test_data = [test_embeddings_input, test_number_of_tokens, test_labels]

    model.to(device)

    train_losses, test_losses = training_loop(data,
                                              test_data, 
                                              model, 
                                              device, 
                                              optimizer, 
                                              loss_fn, 
                                              epochs=epochs, 
                                              batch_size=batch_size, 
                                              return_losses=True
                                            )

    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.savefig('/home/pfytas/losses.png')
    model_path = LSTMEmbeddingLoader.DATA_ROOT / 'lstm_att_classifier'
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(model, model_path / 'model.pt')
