import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import lime
import numpy as np
from lime.lime_text import LimeTextExplainer
from DataLoader import PerReviewDataLoader
import pickle


clf_to_explain = 'per_review_classifier'

num_features = 20
# increasing number of features -> more words per sample -> reduct the mean significance of important words that only appear once
num_samples = 10000
BATCH_SIZE = 200

# paths
path = PerReviewDataLoader.DATA_ROOT / clf_to_explain
model_path = str(path / 'model.pt')


# cuda
device_idx = input("GPU: ")
GPU = True
if GPU:
    device = torch.device("cuda:" + device_idx if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


# Load Data
data_loader = PerReviewDataLoader(device=device,
                                  final_decision='exclude',
                                  allow_empty=False,
                                  truncate_policy='right',
                                  pretrained_weights='scibert_scivocab_uncased',
                                  remove_duplicates=True,
                                  remove_stopwords=False)

data_loader.model.to(device)

text_input = np.array(data_loader.read_reviews_only_text())
embeddings_input = data_loader.read_embeddigns_from_file().to(device)

scores = data_loader.read_scores()
labels = (scores > 5).to(device=device, dtype=torch.float)


valid_size = 0.1

num_train = len(text_input)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))


train_idx, test_idx = indices[split:], indices[:split]

# test set

test_text_input = text_input[test_idx]
test_embeddings_input = embeddings_input[test_idx, :]
test_number_of_reviews = labels[test_idx]
test_labels = labels[test_idx]

## train set
# text_input = text_input[train_idx]
# number_of_reviews = number_of_reviews[train_idx]
# labels = labels[train_idx]

SEP = '\n¬\n'


# load model
model = torch.load(model_path)
model.to(device)

def model_wrapper(perturbations):
    embeddings = []
    cnt = 1
    N = len(perturbations)
    for i in range(0, N, BATCH_SIZE):
        revs = perturbations[i:i+BATCH_SIZE]
        embs = data_loader.reviews_to_embeddings(revs)
        if i % 1000 == 0:
            print(i)
        embeddings.append(embs)
    embeddings = torch.cat(embeddings).to(device)
    preds = model(embeddings, None)
    return torch.cat([1-preds, preds], dim=1).detach().cpu().numpy()




# Create Explainer
class_names = ['rejected', 'accepted']
explainer = LimeTextExplainer(class_names=class_names)
print(test_labels.sum(), len(test_labels))


# Test Accuracy
predictions = model(test_embeddings_input, None)
preds = (predictions.view(-1) >= 0.5).to(device='cpu', dtype=torch.int)
targets = (test_labels >= 0.5).to(device='cpu', dtype=torch.int)

accuracy = (preds == targets).sum() * (1 / len(test_idx))
print('Accuracy on test set: ', accuracy)


# Generate Explanations
idx_to_explain = test_idx
for idx in idx_to_explain:
    exp = explainer.explain_instance(test_text_input[idx], model_wrapper, num_features=num_features, num_samples=num_samples)
    pickle.dump(exp, open(path / ('explanation' + str(idx) + '.p'), 'wb'))
    f = open(path / ('explanation' + str(idx) + '.txt'), "w")
    f.write('Document idx: %d\n' % idx)
    f.write('Probability(accepted): %4f\n' % model_wrapper([test_text_input[idx]])[0, 1])
    f.write('Embedding probability(accepted) :%4f\n' % model(test_embeddings_input[idx:idx+1, :], None))
    f.write('True class: %s\n' % class_names[int(test_labels[idx])])
    f.close()