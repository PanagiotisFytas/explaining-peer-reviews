import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from DataLoader import LSTMPerReviewDataLoader
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
causal_layer = None
causal_layer = 'adversarial'

# aspect = 'CLARITY'
# aspect = None
aspect = 'ORIGINALITY'
# aspect = 'RECOMMENDATION'

data_loader = LSTMPerReviewDataLoader(device=device,
                                      lemmatise=True, 
                                      lowercase=True, 
                                      remove_stopwords=False, 
                                      punctuation_removal=True,
                                      final_decision='exclude',
                                      aspect=aspect,
                                      pretrained_weights='scibert_scivocab_uncased',
                                     )

try:
    embeddings_input = data_loader.read_embeddigns_from_file()
    data_loader.read_reviews_only_text()
except FileNotFoundError:
    # create file with embeddings if it does not exist
    embeddings_input = data_loader.get_embeddings_from_reviews()
    data_loader.write_embeddings_to_file()


number_of_tokens = torch.tensor([review.shape[0] for review in embeddings_input]).to(device)

# number_of_tokens2 = (number_of_tokens>512).to('cpu').numpy()

# number_of_tokens2 = sorted(number_of_tokens2, reverse=True)

# print(number_of_tokens2)
# print(np.count_nonzero(number_of_tokens2))


embeddings_input = rnn.pad_sequence(embeddings_input, batch_first=True)  # pad the reviews to form a tensor
print(embeddings_input.shape)
labels = data_loader.read_labels().to(device)
if causal_layer:
    scores = data_loader.read_aspect_scores().to(device, dtype=torch.float)

_, _, embedding_dimension = embeddings_input.shape

epochs = 110 # 150 # 100 # 110 # 500
batch_size = 150 # 100 # 30
lr = 0.0005 # 0.0001
hidden_dimensions = [128, 64] # [128, 64] # [1500, 700, 300]
lstm_hidden_dimension = 30 # 300 # 500
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
        'cell_type': cell_type,
        'causal_layer': causal_layer
    }
    optimizer = torch.optim.Adam
    lr = lr
    loss_fn = nn.BCELoss
    if not causal_layer:
        data = [embeddings_input, number_of_tokens, labels]
        cross_validation_metrics(network, network_params, optimizer, loss_fn, lr,
                                 epochs, batch_size, device, data, k=5, shuffle=True)
    else:
        data = [embeddings_input, number_of_tokens, labels, scores]
        confounding_loss_fn = nn.MSELoss
        cross_validation_metrics(network, network_params, optimizer, loss_fn, lr,
                                 epochs, batch_size, device, data, confounding_loss_fn=confounding_loss_fn, k=5, shuffle=True)


else:
    # hold-one-out split
    model = LSTMAttentionClassifier(device=device, 
                                    input_size=embedding_dimension,
                                    lstm_hidden_size=lstm_hidden_dimension,
                                    num_layers=num_layers,
                                    bidirectional=bidirectional,
                                    hidden_dimensions=hidden_dimensions,
                                    cell_type=cell_type,
                                    causal_layer=causal_layer
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
    if not causal_layer:
        confounding_loss_fn = None
        data = [embeddings_input, number_of_tokens, labels]
        test_data = [test_embeddings_input, test_number_of_tokens, test_labels]
    else:
        confounding_loss_fn = nn.MSELoss()
        data = [embeddings_input, number_of_tokens, labels, scores]
        test_data = [test_embeddings_input, test_number_of_tokens, test_labels, scores]


    model.to(device)

    losses = training_loop(data,
                           test_data, 
                           model, 
                           device, 
                           optimizer, 
                           loss_fn,
                           confounder_loss_fn=confounding_loss_fn,
                           causal_layer=causal_layer,
                           epochs=epochs, 
                           batch_size=batch_size, 
                           return_losses=True
                         )

    
    if not causal_layer:
        train_losses, test_losses = losses
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.legend()
        plt.savefig('/home/pfytas/losses.png')
        model_path = LSTMPerReviewDataLoader.DATA_ROOT / 'lstm_att_classifier_per_review'
    else:
        train_losses, test_losses, confounding_train_losses, confounding_test_losses = losses
        plt.yscale('log')
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.plot(confounding_train_losses, label='Confounding Train Loss')
        plt.plot(confounding_test_losses, label='Confounding Test Loss')
        plt.legend()
        plt.savefig('/home/pfytas/losses.png')
        model_path = LSTMPerReviewDataLoader.DATA_ROOT / ('lstm_att_classifier_per_review' + causal_layer)
    model_path.mkdir(parents=True, exist_ok=True)
    torch.save(model, model_path / 'model.pt')