import torch
import torch.nn as nn
from DataLoader import PerReviewDataLoader, DataLoader
import numpy as np
from helper_functions import training_loop, cross_validation_metrics
from models import BERTClassifier
import pathlib
import os
import matplotlib.pyplot as plt
import yaml


# read configuration
with open('src/config/BERT_classifier_per_review.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print(config)

# read cude idx
device_idx = config['CUDA']
GPU = True
if GPU:
    device = torch.device("cuda:" + device_idx if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

cross_validation = config['cross_validation']
folds = config['folds']
causal_layer = config['causal_layer']
aspect = config['aspect']

data_loader = PerReviewDataLoader(device=device,
                                  final_decision='exclude',
                                  allow_empty=False,
                                  truncate_policy='right',
                                #   pretrained_weights='bert-base-uncased',
                                  pretrained_weights='scibert_scivocab_uncased',
                                  remove_duplicates=True,
                                  remove_stopwords=False)

try:
    embeddings_input = data_loader.read_embeddigns_from_file()
    data_loader.read_reviews_only_text()
    scores = data_loader.read_scores_from_file()
    labels = (scores > 5).float()
except FileNotFoundError:
    # create file with embeddings if it does not exist
    embeddings_input = data_loader.get_embeddings_from_reviews()
    data_loader.write_embeddings_to_file()
    scores = data_loader.read_scores()
    data_loader.write_scores_to_file()
print(embeddings_input.shape)
print(scores.shape)

_, embedding_dimension = embeddings_input.shape

if causal_layer:
    if aspect == 'abstract':
        confounders = data_loader.read_abstract_embeddings()
        confounders = torch.tensor(data_loader.copy_to_peer_review(confounders), dtype=torch.float)

if causal_layer == 'residual':
    nn_conf = config[causal_layer]
else:
    nn_conf = config['not_residual']


epochs = nn_conf['epochs'] # 200 # 500
batch_size = nn_conf['batch_size'] # 300 # 120
lr = nn_conf['lr'] # 0.0001
hidden_dimensions = nn_conf['hidden_dimensions'] #[128, 64] # [1500, 700, 300] # [700, 300]
causal_hidden_dimensions = nn_conf['causal_hidden_dimensions'] # [64]
BERT_hidden_dimensions = nn_conf['BERT_hidden_dimensions'] # [64]
dropout = nn_conf['dropout']
dropout2 = nn_conf['dropout2']
activation = nn_conf['activation']
activation2 = nn_conf['activation2']


if cross_validation:
    network = BERTClassifier
    network_params = {
        'device': device,
        'input_size': embedding_dimension,
        'hidden_dimensions': hidden_dimensions,
        'causal_hidden_dimensions': causal_hidden_dimensions,
        'BERT_hidden_dimensions': BERT_hidden_dimensions,
        'dropout': dropout,
        'dropout2': dropout2,
        'activation': activation,
        'activation2': activation2,
        'causal_layer': causal_layer
    }
    optimizer = torch.optim.Adam
    lr = lr
    loss_fn = nn.BCELoss
    if not causal_layer:
        data = [embeddings_input, labels, labels]
        cross_validation_metrics(network, network_params, optimizer, loss_fn, lr,
                                 epochs, batch_size, device, data, k=folds, shuffle=True)
    else:
        data = [embeddings_input, labels, labels, confounders]
        confounding_loss_fn = nn.BCELoss
        cross_validation_metrics(network, network_params, optimizer, loss_fn, lr,
                                 epochs, batch_size, device, data, causal_layer=causal_layer,
                                 confounding_loss_fn=confounding_loss_fn, k=folds, 
                                 shuffle=True, loss2_mult=config['loss2_mult'])


else:
    # hold-one-out split
    model = BERTClassifier(device,
                           input_size=embedding_dimension,
                           hidden_dimensions=hidden_dimensions,
                           causal_hidden_dimensions=causal_hidden_dimensions,
                           BERT_hidden_dimensions=BERT_hidden_dimensions,
                           dropout=dropout,
                           dropout2=dropout2,
                           activation=activation,
                           activation2=activation2,
                           causal_layer=causal_layer)
    shuffle = False
    valid_size = 0.1
    print(embeddings_input.shape)

    num_train = embeddings_input.shape[0]
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.shuffle(indices)

    if config['on_validation_set']:
        # will test on validation set and train on train set
        train_idx, test_idx = indices[2*split:], indices[split:2*split]
    else:
        # will test on test set and train on train and vaildation set
        train_idx, test_idx = indices[split:], indices[:split]

    test_embeddings_input = embeddings_input[test_idx, :]
    test_labels = labels[test_idx]

    if causal_layer:
        test_confounders = confounders[test_idx, :]


    embeddings_input = embeddings_input[train_idx, :]
    labels = labels[train_idx]

    if causal_layer:
        confounders = confounders[train_idx, :]


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    if not causal_layer:
        confounding_loss_fn = None
        data = [embeddings_input, labels, labels]
        test_data = [test_embeddings_input, test_labels, test_labels]
    else:
        confounding_loss_fn = nn.BCELoss()
        data = [embeddings_input, labels, labels, confounders]
        test_data = [test_embeddings_input, test_labels, test_labels, test_confounders]

    model.to(device)


    losses = training_loop(data,
                           test_data, 
                           model, 
                           device, 
                           optimizer, 
                           loss_fn,
                           confounder_loss_fn=confounding_loss_fn,
                           verbose=True,
                           causal_layer=causal_layer,
                           epochs=epochs, 
                           batch_size=batch_size, 
                           return_losses=True,
                           loss2_mult=config['loss2_mult']
                         )


    if not causal_layer:
        train_losses, test_losses = losses
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.legend()
        plt.savefig('bert_losses.png')
        model_path = PerReviewDataLoader.DATA_ROOT / 'bert_classifier_per_review'
    else:
        train_losses, test_losses, confounding_train_losses, confounding_test_losses = losses
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.plot(confounding_train_losses, label='Confounding Train Loss')
        plt.plot(confounding_test_losses, label='Confounding Test Loss')
        plt.legend()
        # plt.yscale('log')
        plt.savefig('bert_losses.png')
        model_path = PerReviewDataLoader.DATA_ROOT / ('bert_classifier_per_review' + causal_layer)
    model_path.mkdir(parents=True, exist_ok=True)
    # torch.save(model, model_path / 'model.pt')
