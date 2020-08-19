import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import numpy as np
from helper_functions import training_loop, cross_validation_metrics
from models import LSTMAttentionClassifier
import pathlib
import os
from combine_explanations import LSTMMetrics, cols, combine_explanations
import pandas as pd
import scipy.stats as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import cross_val_predict
from DataLoader import DataLoader
import yaml


def generate_bow_for_lexicon(explanation, reviews, k=100):
    '''
    :param explanation: a dataframe with the global explanations
    :param reviews: preprocessed reviews
    :param k: use the top k words from the lexicon
    :returns a dataframe with the words of a lexicon as columns and a binary vector for each review as row
    '''
    lexicon = explanation['Words'].to_list()[:k]
    bow = pd.DataFrame(columns=lexicon)
    idx = 0
    for paper_reviews in reviews:
        bow_vector = []
        for word in lexicon:
            word_found = 0
            for review in paper_reviews:
                if word in review:
                    word_found = 1
                    break
                else:
                    word_found = 0
            bow_vector.append(word_found)
        bow.loc[idx] = bow_vector
        idx += 1
    return bow



with open('src/config/BERT_classifier_per_review.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


device_idx = config['CUDA']

GPU = True
if GPU:
    device = torch.device("cuda:" + device_idx if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

causal_layer = None
lexicon_size = config['lexicon_size']
# final_decision = 'only'
final_decision = 'exclude'
if final_decision == 'only':
    clf_to_explain = 'final_decision_only'
else:
    clf_to_explain = 'no_final_decision'

# paths
path = DataLoader.DATA_ROOT / clf_to_explain 

# exp = combine_explanations(clf_to_explain=clf_to_explain +  '/second_exp', lemmatize=True)
exp = combine_explanations(clf_to_explain=clf_to_explain, lemmatize=True)
exp['Mean'] = exp['Mean'].abs()
data_loader = DataLoader(device=device,
                                  remove_stopwords=False,
                                  final_decision='exclude',
                                  pretrained_weights='scibert_scivocab_uncased',
                                  allow_empty=False
                                  )

embeddings_input = data_loader.read_embeddigns_from_file()
embeddings_input = rnn.pad_sequence(embeddings_input, batch_first=True).to(device)  # pad the reviews to form a tensor
reviews = data_loader.read_reviews_only_text()

number_of_tokens = torch.tensor([review.shape[0] for review in embeddings_input]).to(device)
labels = data_loader.read_labels().to(device)
aspect = 'abstract'
if aspect == 'abstract':
    confounders = data_loader.read_abstract_embeddings()
elif aspect == 'structure':
    paper_errors, abstract_errors, paper_words, abstract_words = data_loader.read_errors()
    paper_score = paper_errors / paper_words
    abstract_score = abstract_errors / abstract_words
    paper_score = data_loader.copy_to_peer_review(paper_score)
    abstract_score = data_loader.copy_to_peer_review(abstract_score)
    confounders = data_loader.read_handcrafted_features()
    confounders = data_loader.copy_to_peer_review(confounders)
    confounders = np.append(confounders, np.expand_dims(paper_score, axis=1), axis=1)
    confounders = np.append(confounders, np.expand_dims(abstract_score, axis=1), axis=1)
    #confounders = data_loader.read_average_scores(aspect=config['aspect'])
elif aspect == 'grammar':
    paper_errors, abstract_errors, paper_words, abstract_words = data_loader.read_errors()
    paper_score = paper_errors / paper_words
    abstract_score = abstract_errors / abstract_words
    paper_score = data_loader.copy_to_peer_review(paper_score)
    abstract_score = data_loader.copy_to_peer_review(abstract_score)
    confounders = np.concatenate((np.expand_dims(paper_score, axis=1), np.expand_dims(abstract_score, axis=1)), axis=1)


text_input = np.array(reviews)

valid_size = 0.1

num_train = len(text_input)
print('Textinput', text_input.shape)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))


train_idx, test_idx = indices[split:], indices[:split]

# test set

test_text_input = text_input[test_idx]
test_embeddings_input = embeddings_input[test_idx, :, :]
test_number_of_tokens = number_of_tokens[test_idx]
print(labels)
print(labels.shape)
test_labels = labels[test_idx]
test_confounders = confounders[test_idx]

# train set

train_text_input = text_input[train_idx]
print(indices)
train_embeddings_input = embeddings_input[train_idx, :, :]
train_number_of_tokens = number_of_tokens[train_idx]
train_labels = labels[train_idx]
train_confounders = confounders[train_idx]

test_bow = generate_bow_for_lexicon(exp, test_text_input, k=lexicon_size)
test_labels_df = pd.DataFrame(test_labels.view(-1,1).to('cpu').numpy())
train_bow = generate_bow_for_lexicon(exp, train_text_input, k=lexicon_size)
train_labels_df = pd.DataFrame(train_labels.to('cpu').numpy())

if lexicon_size >= 1000:
    train_bow_path = str(path / 'train_bow.csv')
    train_bow.to_csv(train_bow_path, index=False)
    test_bow_path = str(path / 'test_bow.csv')
    test_bow.to_csv(test_bow_path, index=False)


print('###### LR on lexicon (no confounding): ######')
if False:#config['cv_explanation']:
    X = pd.concat([train_bow, test_bow])
    y = pd.concat([train_labels_df, test_labels_df])
    preds = cross_val_predict(LogisticRegression(max_iter=500), X, y, cv=config['folds'])
    print('MSE with labels', mean_squared_error(y, preds))
    print('Classification report:\n', classification_report(y, preds))
else:
    clf =  LogisticRegression(max_iter=1000).fit(train_bow, train_labels_df)
    print(clf.score(test_bow, test_labels_df))
    preds_prob = clf.predict_proba(test_bow)[:, 0]
    print('MSE with probs', mean_squared_error(test_labels_df, preds_prob))
    preds = clf.predict(test_bow)
    print('MSE with labels', mean_squared_error(test_labels_df, preds))
    print('Classification report:\n', classification_report(test_labels_df, preds))
print('#############################################')

print('###### LR on lexicon (confounders conf.): ######')

# concatenate lexicon bag of words with confounders embeddins
train_bow = train_bow.to_numpy()
test_bow = test_bow.to_numpy()
train_confounders = train_confounders
test_confounders = test_confounders
# if causal_layer == 'adversarial':
#     train_confounders = np.expand_dims(train_confounders, axis=1)
#     test_confounders = np.expand_dims(test_confounders, axis=1)
train_bow = np.concatenate((train_bow, train_confounders), axis=1)
test_bow = np.concatenate((test_bow, test_confounders), axis=1)
print(train_bow.shape, test_bow.shape)
if False: #config['cv_explanation']:
    X = np.concatenate([train_bow, test_bow], axis=0)
    y = np.concatenate([train_labels_df, test_labels_df], axis=0)
    preds = cross_val_predict(LogisticRegression(max_iter=500), X, y, cv=config['folds'])
    print('MSE with labels', mean_squared_error(y, preds))
    print('Classification report:\n', classification_report(y, preds))
else:
    clf =  LogisticRegression(max_iter=2000).fit(train_bow, train_labels_df)
    print(clf.score(test_bow, test_labels_df))
    preds_prob = clf.predict_proba(test_bow)[:, 0]
    print('MSE with probs', mean_squared_error(test_labels_df, preds_prob))
    preds = clf.predict(test_bow)
    print('MSE with labels', mean_squared_error(test_labels_df, preds))
    print('Classification report:\n', classification_report(test_labels_df, preds))
    print('#############################################')

print('############## LR on confounders: ##############')

# concatenate lexicon bag of words with confounders embeddins
if False: #config['cv_explanation']:
    X = np.concatenate([train_confounders, test_confounders], axis=0)
    y = np.concatenate([train_labels_df, test_labels_df], axis=0)
    preds = cross_val_predict(LogisticRegression(max_iter=500), X, y, cv=config['folds'])
    print('MSE with labels', mean_squared_error(y, preds))
    print('Classification report:\n', classification_report(y, preds))
else:
    clf =  LogisticRegression(max_iter=1000).fit(train_confounders, train_labels_df)
    print(clf.score(test_confounders, test_labels_df))
    preds_prob = clf.predict_proba(test_confounders)[:, 0]
    print('MSE with probs', mean_squared_error(test_labels_df, preds_prob))
    preds = clf.predict(test_confounders)
    print('MSE with labels', mean_squared_error(test_labels_df, preds))
    print('Classification report:\n', classification_report(test_labels_df, preds))
    print('#############################################')
