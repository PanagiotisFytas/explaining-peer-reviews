import torch
import torch.nn.utils.rnn as rnn
from DataLoader import DataLoader
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.model_selection import cross_val_predict


data_loader = DataLoader(device='cpu',
                         truncate_policy='right',
                         final_decision='only',
                         allow_empty='False',
                         pretrained_weights='scibert_scivocab_uncased',
                         remove_duplicates=True,
                         remove_stopwords=False
                         )

embeddings_input = data_loader.read_embeddigns_from_file()
number_of_reviews = torch.tensor([reviews.shape[0] for reviews in embeddings_input])
embeddings_input = rnn.pad_sequence(embeddings_input, batch_first=True).numpy()# pad the reviews to form a tensor
print(embeddings_input.shape)
labels = data_loader.read_labels().numpy()


majoriy_clf = DummyClassifier(strategy='most_frequent')


preds = cross_val_predict(majoriy_clf, embeddings_input, labels, cv=5)

print('5-CV Majority Classifier:\n', classification_report(labels, preds, output_dict=True))


valid_size = 0.1

num_train = embeddings_input.shape[0]
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))


train_idx, test_idx = indices[split:], indices[:split]

test_embeddings_input = embeddings_input[test_idx, :, :]
test_number_of_reviews = number_of_reviews[test_idx]
test_labels = labels[test_idx]

embeddings_input = embeddings_input[train_idx, :, :]
number_of_reviews = number_of_reviews[train_idx]
labels = labels[train_idx]

majoriy_clf.fit(embeddings_input, labels)

preds = majoriy_clf.predict(test_embeddings_input)

print('Majority Classifier:\n', classification_report(test_labels, preds, output_dict=True))

print(mean_squared_error(test_labels, preds))