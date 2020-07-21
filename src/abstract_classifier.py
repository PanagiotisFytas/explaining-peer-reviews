import torch
import torch.nn as nn
from DataLoader import DataLoader
import numpy as np
from helper_functions import training_loop_scores, cross_validation_metrics_scores
from models import AbstractClassifier


device_idx = input("GPU: ")
GPU = True
if GPU:
    device = torch.device("cuda:" + device_idx if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

cross_validation = False
# cross_validation = True


data_loader = DataLoader(device=device,
                        final_decision='only',
                        allow_empty=False,
                        truncate_policy='right',
                        pretrained_weights='scibert_scivocab_uncased',
                        remove_duplicates=True,
                        remove_stopwords=False)


embeddings_input = data_loader.read_abstract_embeddings()
labels = data_loader.read_labels().to(device, dtype=torch.float)

print(embeddings_input.shape)
print(labels.shape)

embeddings_input = embeddings_input.to(device)

_, embedding_dimension = embeddings_input.shape

epochs = 200  # 500
batch_size = 300  # 300
lr = 0.0001
hidden_dimensions = [64] # [128, 64] # [1500, 700, 300]


if cross_validation:
    network = AbstractClassifier
    network_params = {
        'input_size': embedding_dimension,
        'hidden_dimensions': hidden_dimensions,
    }
    optimizer = torch.optim.Adam
    lr = lr
    loss_fn = nn.BCELoss
    data = [embeddings_input, labels, labels]
    cross_validation_metrics_scores(network, network_params, optimizer, loss_fn, lr,
                                    epochs, batch_size, device, data, k=5, shuffle=True)
    # # dataset = CustomDataset(embeddings_input, number_of_reviews, labels)
    # dataset = Dataset({'inp': embeddings_input, 'lengths': number_of_reviews}, labels)
    # # X_dict = {'inp': embeddings_input, 'lengths': number_of_reviews}
    # print(embeddings_input.shape, number_of_reviews.shape, labels.shape)
    # net.fit(dataset, y=labels)
    # preds = cross_val_predict(net, dataset, y=labels.to('cpu'), cv=5)
else:
    # hold-one-out split
    model = AbstractClassifier(input_size=embedding_dimension,
                               hidden_dimensions=hidden_dimensions)
    shuffle = False
    valid_size = 0.1
    print(embeddings_input.shape)

    num_train = embeddings_input.shape[0]
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]

    test_embeddings_input = embeddings_input[test_idx, :]
    test_labels = labels[test_idx]

    embeddings_input = embeddings_input[train_idx, :]
    labels = labels[train_idx]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    data = [embeddings_input, labels, labels]
    test_data = [test_embeddings_input, test_labels, test_labels]

    model.to(device)

    training_loop_scores(data, test_data, model, device, optimizer, loss_fn, epochs=epochs, batch_size=batch_size)

    model_path = DataLoader.DATA_ROOT / 'abstract_classifier'
    model_path.mkdir(parents=True, exist_ok=True)

    torch.save(model, model_path / 'model.pt')
