import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from DataLoader import DataLoader
import numpy as np
from helper_functions import training_loop, cross_validation_metrics
from models import MultiheadClassifier
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


data_loader = DataLoader(device=device,
                         final_decision='exclude',
                         allow_empty=False,
                         truncate_policy='right',
                         pretrained_weights='scibert_scivocab_uncased',
                         remove_duplicates=True,
                         remove_stopwords=False)

try:
    embeddings_input = data_loader.read_embeddigns_from_file()
except FileNotFoundError:
    # create file with embeddings if it does not exist
    embeddings_input = data_loader.get_embeddings_from_reviews()
    data_loader.write_embeddings_to_file()
#
# for idx, reviews in enumerate(embeddings_input):
#     duplicates_removed = []
#     for review in reviews:
#         duplicate = False
#         for r in duplicates_removed:
#             if torch.all(r.eq(review)):
#                 duplicate = True
#                 break
#         if not duplicate:
#             duplicates_removed.append(review)
#     embeddings_input[idx] = torch.stack(duplicates_removed)


number_of_reviews = torch.tensor([reviews.shape[0] for reviews in embeddings_input]).to(device)
embeddings_input = rnn.pad_sequence(embeddings_input, batch_first=True).to(device)  # pad the reviews to form a tensor
print(embeddings_input.shape)
labels = data_loader.read_labels().to(device)

_, _, embedding_dimension = embeddings_input.shape

epochs = 200
batch_size = 100  # 100
lr = 0.0001
hidden_dimensions = [128, 64] # [1500, 700, 300]
heads = 2

if cross_validation:
    network = MultiheadClassifier
    network_params = {
        'input_size': embedding_dimension,
        'hidden_dimensions': hidden_dimensions,
        'heads': heads
    }
    optimizer = torch.optim.Adam
    lr = lr
    loss_fn = nn.BCELoss
    data = [embeddings_input, number_of_reviews, labels]
    cross_validation_metrics(network, network_params, optimizer, loss_fn, lr,
                             epochs, batch_size, device, data, k=10, shuffle=True)
    # # dataset = CustomDataset(embeddings_input, number_of_reviews, labels)
    # dataset = Dataset({'inp': embeddings_input, 'lengths': number_of_reviews}, labels)
    # # X_dict = {'inp': embeddings_input, 'lengths': number_of_reviews}
    # print(embeddings_input.shape, number_of_reviews.shape, labels.shape)
    # net.fit(dataset, y=labels)
    # preds = cross_val_predict(net, dataset, y=labels.to('cpu'), cv=5)
else:
    # hold-one-out split
    model = MultiheadClassifier(input_size=embedding_dimension,
                                hidden_dimensions=hidden_dimensions,
                                heads=heads)
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

    losses = training_loop(data, test_data, model, device, optimizer, loss_fn, 
                           return_losses=True, epochs=epochs, batch_size=batch_size)

    train_losses, test_losses = losses
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.legend()
    plt.savefig('/home/pfytas/peer-review-classification/bert_final_decision_losses.png')


    model_path = DataLoader.DATA_ROOT / 'no_final_decision'
    model_path.mkdir(parents=True, exist_ok=True)

    # torch.save(model, model_path / 'model.pt')
