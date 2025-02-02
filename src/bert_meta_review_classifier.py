import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from DataLoader import DataLoader
import numpy as np
from helper_functions import training_loop, cross_validation_metrics
from models import AttentionClassifier
import pathlib
import os
import matplotlib.pyplot as plt


#ask for which GPU  to use
device_idx = input("Specify GPU Index: ")
GPU = True
if GPU:
    device = torch.device("cuda:" + device_idx if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

#specify whether to cross validate
cross_validation = False
# cross_validation = True


data_loader = DataLoader(device=device,
                         truncate_policy='right',
                         final_decision='only',
                         allow_empty='False',
                         pretrained_weights='scibert_scivocab_uncased',
                         remove_duplicates=True,
                         remove_stopwords=False
                         )

try:
    # try and read the embeddings from disk
    embeddings_input = data_loader.read_embeddigns_from_file()
except FileNotFoundError:
    # create file with embeddings if it does not exist
    embeddings_input = data_loader.get_embeddings_from_reviews()
    data_loader.write_embeddings_to_file()


number_of_reviews = torch.tensor([reviews.shape[0] for reviews in embeddings_input]).to(device)
embeddings_input = rnn.pad_sequence(embeddings_input, batch_first=True).to(device)  # pad the reviews to form a tensor
print(embeddings_input.shape)
labels = data_loader.read_labels().to(device)

_, _, embedding_dimension = embeddings_input.shape

#specify hyperparams
epochs = 100
batch_size = 100  # 30
lr = 0.0001
hidden_dimensions = [128, 64] # [1500, 700, 300]

if cross_validation:
    network = AttentionClassifier
    network_params = {
        'input_size': embedding_dimension,
        'hidden_dimensions': hidden_dimensions,
    }
    optimizer = torch.optim.Adam
    lr = lr
    loss_fn = nn.BCELoss
    data = [embeddings_input, number_of_reviews, labels]
    cross_validation_metrics(network, network_params, optimizer, loss_fn, lr,
                             epochs, batch_size, device, data, k=5, shuffle=True)
else:
    # hold-one-out split
    model = AttentionClassifier(dropout=0.5, input_size=embedding_dimension, hidden_dimensions=hidden_dimensions)
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
    test_number_of_reviews = number_of_reviews[test_idx]
    test_labels = labels[test_idx]

    embeddings_input = embeddings_input[train_idx, :, :]
    number_of_reviews = number_of_reviews[train_idx]
    labels = labels[train_idx]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    data = [embeddings_input, number_of_reviews, labels]
    test_data = [test_embeddings_input, test_number_of_reviews, test_labels]

    model.to(device)

    # train the model
    losses = training_loop(data, test_data, model, device, optimizer, loss_fn, return_losses=True, epochs=epochs, batch_size=batch_size)

    train_losses, test_losses = losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.savefig('bert_meta_losses.png')

    model_path = DataLoader.DATA_ROOT / 'final_decision_only'
    model_path.mkdir(parents=True, exist_ok=True)

    # save model
    torch.save(model, model_path / 'model.pt')
