from DataLoader import LSTMPerReviewDataLoader, LSTMEmbeddingLoader
import pathlib
import os
import pandas as pd
import scipy.stats as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import cross_val_predict
from scipy.integrate import simps
import numpy as np
import matplotlib.pyplot as plt


per_review = True # False
causal_layer = None
lr_steps = range(10, 501, 10) #50)

if per_review:
    clfs = ['lstm_att_classifier_per_review', 'bert_classifier_per_review', 
    'no_final_decision',  'bow_classifier_per_review']
    plt_labels = ['GRU+ATTN', 'BERT+lime', 'BERT+lime final acceptance', 'BoW']
    # plt_labels = ['GRU+ATTN', 'BERT+lime individual', 'BERT+lime final acceptance']
    points = ['.','.','.']
    title = 'Peer Reviews'
    data_loader = LSTMPerReviewDataLoader(device='cpu',
                                          lemmatise=True, 
                                          lowercase=True, 
                                          remove_stopwords=False, 
                                          punctuation_removal=True,
                                          final_decision='exclude',
                                          pretrained_weights='scibert_scivocab_uncased',
                                          )
    _reviews = data_loader.read_reviews_only_text()
    labels = data_loader.read_labels().numpy()
    data_loader2 = LSTMEmbeddingLoader(device='cpu',
                                    lemmatise=True, 
                                    lowercase=True, 
                                    remove_stopwords=False, 
                                    punctuation_removal=True,
                                    final_decision='exclude',
                                    pretrained_weights='scibert_scivocab_uncased',
                                    ) 
    final_labels = data_loader2.read_labels().numpy()
else:
    clfs = ['lstm_att_classifier', 'final_decision_only']
    plt_labels = ['GRU+ATTN', 'BERT+lime']
    points = ['.','.']
    title = 'Meta-reviews'
    data_loader = LSTMEmbeddingLoader(device='cpu',
                                    lemmatise=True, 
                                    lowercase=True, 
                                    remove_stopwords=False, 
                                    punctuation_removal=True,
                                    final_decision='only',
                                    pretrained_weights='scibert_scivocab_uncased',
                                    )
    labels = data_loader.read_labels().numpy()


if causal_layer:
    clfs = [clf + 'residual' for clf in clfs]


valid_size = 0.1
num_train = len(labels)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))
train_idx, test_idx = indices[split:], indices[:split]

test_labels = labels[test_idx]
train_labels = labels[train_idx]

if per_review:
    final_num_train = len(final_labels)
    final_indices = list(range(final_num_train))
    final_split = int(np.floor(valid_size * final_num_train))
    final_train_idx, final_test_idx = final_indices[final_split:], final_indices[:final_split]
    test_final_labels = final_labels[final_test_idx]
    train_final_labels = final_labels[final_train_idx]


clfs_losses = []
clfs_probs1_losses = []
clfs_probs2_losses = []
clfs_accs = []
for clf in clfs:
    print(clf)
    train_bow_path = str(data_loader.DATA_ROOT / clf / 'train_bow.csv')
    train_bow = pd.read_csv(train_bow_path)
    test_bow_path = str(data_loader.DATA_ROOT / clf / 'test_bow.csv')
    test_bow = pd.read_csv(test_bow_path)
    losses = []
    probs1_losses = []
    probs2_losses = []
    accs = []
    if clf == 'no_final_decision':
        test_labels_to_use = test_final_labels
        train_labels_to_use = train_final_labels
    else:
        test_labels_to_use = test_labels
        train_labels_to_use = train_labels
    for lexicon_size in lr_steps:
        print(lexicon_size)
        clf =  LogisticRegression(max_iter=5000).fit(train_bow.iloc[:, :lexicon_size], train_labels_to_use)
        preds_prob1 = clf.predict_proba(test_bow.iloc[:, :lexicon_size])[:, 0]
        probs1_losses.append(mean_squared_error(1-test_labels_to_use, preds_prob1))
        preds_prob2 = clf.predict_proba(test_bow.iloc[:, :lexicon_size])[:, 1]
        probs2_losses.append(mean_squared_error(test_labels_to_use, preds_prob2))
        preds = clf.predict(test_bow.iloc[:, :lexicon_size])
        losses.append(mean_squared_error(test_labels_to_use, preds))
        output_dict = classification_report(test_labels_to_use, preds, output_dict=True)
        accs.append(output_dict['macro avg']['f1-score'])
    
    clfs_losses.append(losses)
    clfs_probs1_losses.append(probs1_losses)
    clfs_probs2_losses.append(probs2_losses)
    clfs_accs.append(accs)

for idx in range(len(clfs)):
    plt.plot(lr_steps, clfs_losses[idx], marker='.', label=plt_labels[idx])
plt.legend()
plt.title(title)
plt.xlabel('Lexicon Size')
plt.ylabel('Loss')
if per_review:
    if not causal_layer:
        plt.savefig('/home/pfytas/peer-review-classification/clfs_losses_per_review.png')
    else:
        plt.savefig('/home/pfytas/peer-review-classification/clfs_losses_residual.png')
else:
    plt.savefig('/home/pfytas/peer-review-classification/clfs_losses_meta.png')
plt.clf()

for idx in range(len(clfs)):
    plt.plot(lr_steps, clfs_probs1_losses[idx], marker='.', label=plt_labels[idx])
plt.legend()
plt.title(title)
plt.xlabel('Lexicon Size')
plt.ylabel('Loss')
if per_review:
    if not causal_layer:
        plt.savefig('/home/pfytas/peer-review-classification/clfs_probs1_losses_per_review.png')
    else:
        plt.savefig('/home/pfytas/peer-review-classification/clfs_probs1_losses_residual.png')
else:
    plt.savefig('/home/pfytas/peer-review-classification/clfs_probs1_losses_meta.png')
plt.clf()

for idx in range(len(clfs)):
    plt.plot(lr_steps, clfs_probs2_losses[idx], marker='.', label=plt_labels[idx])
plt.legend()
plt.title(title)
plt.xlabel('Lexicon Size')
plt.ylabel('Loss')
if per_review:
    if not causal_layer:
        plt.savefig('/home/pfytas/peer-review-classification/clfs_probs2_losses_per_review.png')
    else:
        plt.savefig('/home/pfytas/peer-review-classification/clfs_probs2_losses_residual.png')
else:
    plt.savefig('/home/pfytas/peer-review-classification/clfs_probs2_losses_meta.png')
plt.clf()

for idx in range(len(clfs)):
    plt.plot(lr_steps, clfs_accs[idx], marker='.', label=plt_labels[idx])
plt.legend()
plt.title(title)
plt.xlabel('Lexicon Size')
plt.ylabel('F1')
if per_review:
    if not causal_layer:
        plt.savefig('/home/pfytas/peer-review-classification/clfs_accs_per_review.png')
    else:
        plt.savefig('/home/pfytas/peer-review-classification/clfs_accs_residual.png')
else:
    plt.savefig('/home/pfytas/peer-review-classification/clfs_accs_meta.png')
plt.clf()


print('Area Under Curve with Simpson\' Rule (for F1)')

for idx in range(len(clfs)):
    auc = simps(clfs_accs[idx])
    print(plt_labels[idx], ' : ', auc)
