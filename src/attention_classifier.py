import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from DataLoader import DataLoader
import numpy as np
from helper_functions import training_loop, cross_validation_metrics
from models import AttentionClassifier


GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

cross_validation = True

data_loader = DataLoader(device=device, truncate_policy='right')

embeddings_input = data_loader.read_embeddigns_from_file()
number_of_reviews = torch.tensor([reviews.shape[0] for reviews in embeddings_input]).to(device)
embeddings_input = rnn.pad_sequence(embeddings_input, batch_first=True).to(device)  # pad the reviews to form a tensor
print(embeddings_input.shape)
labels = data_loader.read_labels().to(device)

_, _, embedding_dimension = embeddings_input.shape

epochs = 500
batch_size = 100  # 30
lr = 0.0001
hidden_dimensions = [1500, 700, 300]

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
    # # dataset = CustomDataset(embeddings_input, number_of_reviews, labels)
    # dataset = Dataset({'inp': embeddings_input, 'lengths': number_of_reviews}, labels)
    # # X_dict = {'inp': embeddings_input, 'lengths': number_of_reviews}
    # print(embeddings_input.shape, number_of_reviews.shape, labels.shape)
    # net.fit(dataset, y=labels)
    # preds = cross_val_predict(net, dataset, y=labels.to('cpu'), cv=5)
else:
    # hold-one-out split
    model = AttentionClassifier(input_size=embedding_dimension, hidden_dimensions=hidden_dimensions)
    shuffle = False
    valid_size = 0.2
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

    training_loop(data, test_data, model, device, optimizer, loss_fn, epochs=epochs, batch_size=batch_size)

    torch.save(model, 'attention_classifier.pt')
