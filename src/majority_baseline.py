import torch
import torch.nn.utils.rnn as rnn
from DataLoader import DataLoader
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict


data_loader = DataLoader(device='cpu', truncate_policy='right')

embeddings_input = data_loader.read_embeddigns_from_file()
number_of_reviews = torch.tensor([reviews.shape[0] for reviews in embeddings_input])
embeddings_input = rnn.pad_sequence(embeddings_input, batch_first=True).numpy()# pad the reviews to form a tensor
print(embeddings_input.shape)
labels = data_loader.read_labels().numpy()


majoriy_clf = DummyClassifier(strategy='most_frequent')

preds = cross_val_predict(majoriy_clf, embeddings_input, labels)
print('Majority Classifier:\n', classification_report(labels, preds))