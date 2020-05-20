# -*- coding: utf-8 -*-
"""lime_final_decision.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ptXKhRBC2WrfORS2fkEKoY0fGT7fDV6I
"""

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import lime
import os
import numpy as np
from lime.lime_text import LimeTextExplainer
from DataLoader import DataLoader
import matplotlib.pyplot as plt

GPU = True
device_idx = 0
if GPU:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

data_loader = DataLoader(device=device,
                         truncate_policy='right',
                         final_decision='only',
                         allow_empty='False',
                         pretrained_weights='scibert_scivocab_uncased',
                         remove_duplicates=True
                         )

data_loader.model.to(device)

text_input = np.array(data_loader.read_reviews_only_text())
embeddings_input = data_loader.read_embeddigns_from_file()
number_of_reviews = torch.tensor([len(reviews) for reviews in text_input]).to(device)
emb_number_of_reviews = torch.tensor([len(reviews) for reviews in embeddings_input]).to(device)
embeddings_input = rnn.pad_sequence(embeddings_input, batch_first=True).to(device)  # pad the reviews to form a tensor
labels = data_loader.read_labels().to(device)


print(text_input[0][0])

valid_size = 0.2

num_train = len(text_input)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))


train_idx, test_idx = indices[split:], indices[:split]

# test set

test_text_input = text_input[test_idx]
test_embeddings_input = embeddings_input[test_idx, :, :]
test_number_of_reviews = number_of_reviews[test_idx]
test_labels = labels[test_idx]

## train set
# text_input = text_input[train_idx]
# number_of_reviews = number_of_reviews[train_idx]
# labels = labels[train_idx]
test_text_input[0]

SEP = ' ¬ '

model = torch.load('final_decision_only.pt')
model.to(device)

def model_wrapper(perturbations):
    embeddings = []
    cnt = 1
    for perturbation in perturbations:
        if cnt % 1000 == 0:
            print(cnt)
        cnt += 1
        rev = perturbation.split(SEP)
#         print(reviews)
        embeddings.append(data_loader.reviews_to_embeddings(rev))
#         print(embeddings[cnt-2].shape)
#     print(len(embeddings))
    embeddings.append(embeddings_input[0])
#     number_of_reviews = torch.tensor([reviews.shape[0] for reviews in embeddings]).to(device)
    embeddings = rnn.pad_sequence(embeddings, batch_first=True).to(device)  # pad the reviews to form a tensor
    embeddings = embeddings[:-1]
    preds = model(embeddings, None)
    return torch.cat([1-preds, preds], dim=1).detach().cpu().numpy()


# takes a list of reviews and return them in a single string joined by the seperator ` ¬ `
def join_reviews(reviews):
    return SEP.join(reviews)

class_names = ['rejected', 'accepted']
explainer = LimeTextExplainer(class_names=class_names)
print(test_labels.sum(), len(test_labels))

predictions = model(test_embeddings_input, None)
preds = (predictions.view(-1) >= 0.5).to(device='cpu', dtype=torch.int)
targets = (test_labels >= 0.5).to(device='cpu', dtype=torch.int)

accuracy = (preds == targets).sum() * (1 / 85)
print('Accuracy on test set: ', accuracy)



# idx = 20
# exp = explainer.explain_instance(join_reviews(test_text_input[idx]), model_wrapper, num_features=50, num_samples=1000)
# print('Document id: %d' % idx)
# print('Probability(accepted) =', model_wrapper([join_reviews(test_text_input[idx])])[0, 1])
# print('Embedding probability(accepted) =', model(test_embeddings_input[idx:idx+1, :, :], None))
# print('True class: %s' % class_names[int(test_labels[idx])])
#
# # exp.as_list()
#
# # Commented out IPython magic to ensure Python compatibility.
# # %matplotlib inline
# # fig = exp.as_pyplot_figure()
#
# # 1000 perturbations, idx = 2
# exp.show_in_notebook(text=True)
#
# idx = 20
# exp = explainer.explain_instance(join_reviews(test_text_input[idx]), model_wrapper, num_features=10, num_samples=10000)
# print('Document id: %d' % idx)
# print('Probability(accepted) =', model_wrapper([join_reviews(test_text_input[idx])])[0, 1])
# print('Embedding probability(accepted) =', model(test_embeddings_input[idx:idx+1, :, :], None))
# print('True class: %s' % class_names[int(test_labels[idx])])
#
# exp.show_in_notebook(text=True)
#
# idx = 20
# exp = explainer.explain_instance(join_reviews(test_text_input[idx]), model_wrapper, num_features=10, num_samples=5000)
# print('Document id: %d' % idx)
# print('Probability(accepted) =', model_wrapper([join_reviews(test_text_input[idx])])[0, 1])
# print('Embedding probability(accepted) =', model(test_embeddings_input[idx:idx+1, :, :], None))
# print('True class: %s' % class_names[int(test_labels[idx])])
#
# exp.show_in_notebook(text=True)
name = os.path.basename(__file__).rstrip('.py')
explanations = []
for idx in range(5):
    exp = explainer.explain_instance(join_reviews(test_text_input[idx]), model_wrapper, num_features=50, num_samples=2000)
    print('Document id: %d' % idx)
    print('Probability(accepted) =', model_wrapper([join_reviews(test_text_input[idx])])[0, 1])
    print('Embedding probability(accepted) =', model(test_embeddings_input[idx:idx+1, :, :], None))
    print('True class: %s' % class_names[int(test_labels[idx])])
    exp.save_to_file(name + str(idx) + '.out')
    explanations.append(dict(exp.as_list()))

avg_dict = {}
for exp in explanations:
    for word, value in exp.items():
        if word not in avg_dict:
            avg_dict[word] = (value, 1)
        else:
            acc, cnt = avg_dict[word]
            avg_dict[word] = (acc+value, cnt+1)

for word, value in avg_dict.items():
    avg_dict[word] = value[0]/value[1]

print({k: v for k, v in sorted(avg_dict.items(), key=lambda item: abs(item[1]))})