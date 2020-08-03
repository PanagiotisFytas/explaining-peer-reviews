from DataLoader import LSTMPerReviewDataLoader
import pandas as pd
from nltk import ngrams
from nltk.corpus import stopwords
import pathlib
import os
import matplotlib.pyplot as plt
import yaml
import numpy as np
import torch


try:
    stopwords_en = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    stopwords_en = stopwords.words('english')


def filter_ngrams(n_grams, lexicon_words):
    # lexicon_words = lexicon['Words'].tolist()
    filtered_ngrams = []
    for n_gram in n_grams:
        stopword_cnt = 0
        word_in_lexicon = False
        for word in n_gram:
            if not word_in_lexicon and word in lexicon_words:
                # if word_in_lexicon is already true dont keep searching for words in lexicon
                word_in_lexicon = True
            if word in stopwords_en:
                stopword_cnt += 1
        if stopword_cnt < len(n_gram) and word_in_lexicon:
            # only take the ngram if not every word is a stopword and exist one word in the lexicon
            filtered_ngrams.append(n_gram)
    return filtered_ngrams


def generate_freq_bow_for_lexicon(lexicon, word_to_idx, reviews, grams=(2,3,4)):
    '''
    :param lexicon: a list with the ngrams in the lexicon
    :param reviews: preprocessed reviews
    :param train: if True with remove from the lexicon rare n-grams
    :param filter_threshold=k: filter ngrams occuring in k or less samples
    :returns a dataframe with the words of a lexicon as columns and a binary vector for each review as row
    '''
    # lexicon = explanation['Words'].to_list()
    bow = pd.DataFrame(columns=lexicon)
    idx = 0
    L = len(lexicon)
    for review in reviews:
        print(idx)
        bow_vector = np.zeros(L)
        n_grams = {}
        for n in grams:
            n_grams[n] = list(ngrams(review, n))
        N = len(review)
        for word in review:
            if word in word_to_idx:
                bow_vector[word_to_idx[word]] = review.count(word) /  N
        for n in grams:
            for n_gram in n_grams[n]:
                if n_gram in word_to_idx:
                    bow_vector[word_to_idx[n_gram]] = n_grams[n].count(n_gram) / (N-n+1)
        bow.loc[idx] = bow_vector
        idx += 1
    return bow


def get_word_to_idx(lexicon):
    word_to_idx = {}
    idx = 0
    for n_gram in lexicon:
        word_to_idx[n_gram] = idx
        idx += 1
    return word_to_idx

def filter_lexicon(lexicon, reviews, filter_threshold=4, grams=(2,3,4)):
    '''
    :param lexicon: a list with the ngrams in the lexicon
    :param reviews: preprocessed reviews
    :param filter_threshold=k: filter ngrams occuring in k or less samples
    :returns a dataframe with the words of a lexicon as columns and a binary vector for each review as row
    '''
    # lexicon = explanation['Words'].to_list()
    word_to_idx = get_word_to_idx(lexicon)
    bow = generate_freq_bow_for_lexicon(lexicon, word_to_idx, reviews, grams=grams)
    non_zero_counts = bow.astype(bool).sum(axis=0)
    return non_zero_counts.index[non_zero_counts > filter_threshold].tolist()


with open('src/config/lstm_att_classifier_per_review.yaml') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

print(config)

# device_idx = input("GPU: ")
device_idx = config['CUDA']
GPU = True
if GPU:
    device = torch.device("cuda:" + device_idx if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)

cross_validation = config['cross_validation']
folds = config['folds']
# cross_validation = True

# causal_layer = None
# causal_layer = 'adversarial'
causal_layer = config['causal_layer']

# aspect = 'CLARITY'
# aspect = 'ORIGINALITY'
aspect = config['aspect']


task = config['task']

data_loader = LSTMPerReviewDataLoader(device=device,
                                      lemmatise=True, 
                                      lowercase=True, 
                                      remove_stopwords=False, 
                                      punctuation_removal=True,
                                      final_decision='exclude',
                                      aspect='RECOMMENDATION',
                                      pretrained_weights='scibert_scivocab_uncased',
                                     )

text_input = data_loader.read_reviews_only_text()
text_input = np.array(data_loader.preprocessor.preprocess(text_input))
labels = data_loader.read_labels(task=task)

if causal_layer:
    if aspect == 'structure':
        raise NotImplementedError()
    elif aspect == 'abstract':
        confounders = data_loader.read_abstract_embeddings()
        confounders = torch.tensor(data_loader.copy_to_peer_review(confounders), dtype=torch.float)
        adversarial_out = None
    elif aspect == 'grammar_errors':
        raise NotImplementedError()
else:
    adversarial_out = None


valid_size = 0.1

num_train = len(text_input)
print('Textinput', text_input.shape)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))


if config['on_validation_set']:
    # will test on validation set and train on train set
    train_idx, test_idx = indices[2*split:], indices[split:2*split]
else:
    # will test on test set and train on train and vaildation set
    train_idx, test_idx = indices[split:], indices[:split]

# test set

test_text_input = text_input[test_idx]
test_labels = labels[test_idx]
if causal_layer:
    test_confounders = confounders[test_idx]

# train set

train_text_input = text_input[train_idx]
train_labels = labels[train_idx]
if causal_layer:
    train_confounders = confounders[train_idx]


# read lexicon
causal_layer = config['causal_layer']
lexicon_size = config['lexicon_size']
if not causal_layer:
    clf_to_explain = 'lstm_att_classifier_per_review'
else:
    clf_to_explain = 'lstm_att_classifier_per_review' + causal_layer
path = LSTMPerReviewDataLoader.DATA_ROOT / clf_to_explain
lexicon_path = str(path / 'lexicon.csv')
lexicon = pd.read_csv(lexicon_path)
lexicon = lexicon.head(lexicon_size)['Words'].tolist()

# r0 = pd.Series(ngrams(reviews[0], 2), index=None)
# r1 = pd.Series(ngrams(reviews[1], 2), index=None)
# print(list(ngrams(reviews[0], 2)))
# print(r0)
# print(r0.value_counts())
# print(r1.value_counts())
# print(r0.add(r1, fill_value=0).value_counts())
# print(r1.add(r0, fill_value=0).value_counts())
# print(r0.append(r1).value_counts())
# print(r1.append(r0).value_counts())
# print(pd.Series(ngrams(reviews[0], 2)))
unigrams = pd.Series(lexicon)
bigrams = pd.Series()
for review in train_text_input:
    bigrams = bigrams.append(pd.Series(ngrams(review, 2)))
bigrams = pd.Series(filter_ngrams(bigrams.drop_duplicates(), lexicon))

trigrams = pd.Series()
for review in train_text_input:
    trigrams = trigrams.append(pd.Series(ngrams(review, 3)))
trigrams = pd.Series(filter_ngrams(trigrams.drop_duplicates(), lexicon))

quadgrams = pd.Series()
for review in train_text_input:
    quadgrams = quadgrams.append(pd.Series(ngrams(review, 3)))
quadgrams = pd.Series(filter_ngrams(quadgrams.drop_duplicates(), lexicon))

n_grams = unigrams.append(bigrams).append(trigrams).append(quadgrams)
print(n_grams)

n_gram_lexicon = n_grams


# filtered_bow, filtered_lexicon = generate_freq_bow_for_lexicon(n_gram_lexicon, word_to_idx, test_text_input)
filtered_lexicon = filter_lexicon(n_gram_lexicon, train_text_input)
# print(filtered_bow)
print(n_gram_lexicon, len(n_gram_lexicon))
print(filtered_lexicon, len(filtered_lexicon))