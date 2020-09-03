import torch
from DataLoader import PerReviewDataLoader
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_predict


data_loader = PerReviewDataLoader(device='cpu',
                                  final_decision='exclude',
                                  allow_empty=False,
                                  truncate_policy='right',
                                  pretrained_weights='scibert_scivocab_uncased',
                                  remove_duplicates=True,
                                  remove_stopwords=False)

# try:
embeddings_input = data_loader.read_embeddigns_from_file()
scores = data_loader.read_scores_from_file()

labels = (scores > 5).to(device='cpu', dtype=torch.int)

print(embeddings_input.shape)
print(labels.shape)

majoriy_clf = DummyClassifier(strategy='most_frequent')


preds = cross_val_predict(majoriy_clf, embeddings_input, labels, cv=10)

print('10-CV Majority Classifier:\n', classification_report(labels, preds, output_dict=True))


valid_size = 0.1

num_train = embeddings_input.shape[0]
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))


train_idx, test_idx = indices[split:], indices[:split]

test_embeddings_input = embeddings_input[test_idx, :]
test_labels = labels[test_idx]

embeddings_input = embeddings_input[train_idx, :]
labels = labels[train_idx]

majoriy_clf.fit(embeddings_input, labels)

preds = majoriy_clf.predict(test_embeddings_input)

print('Majority Classifier:\n', classification_report(test_labels, preds, digits=4))

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

scores = scores.to(device='cpu').numpy()
test_scores = scores[test_idx]

plt.hist([scores, test_scores],  bins=range(1,12), histtype='bar', label=['Overall Scores', 'Test Set Scores'],
         weights=[np.ones(len(scores)) / len(scores), np.ones(len(test_scores)) / len(test_scores)],
         align='left')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
plt.legend()
plt.xlabel('Recommendation Score')
plt.xticks(range(1,11))
plt.title('Review Recommendation Score Distribution')
plt.savefig('/home/pfytas/peer-review-classification/hist_scores.png')
