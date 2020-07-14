import pickle
import pandas as pd
from DataLoader import DataLoader
import os
from nltk.stem import WordNetLemmatizer
import numpy as np
from helper_functions import natural_sort_key
import re
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


lemmatizer = WordNetLemmatizer()

cols = ['Words', 'Mean', '#Total', '#Positive', '#Neg', 'PosMean', 'NegMean']
final_decision = 'only'
# final_decision = 'exclude'
if final_decision == 'only':
    clf_to_explain = 'final_decision_only'
else:
    clf_to_explain = 'no_final_decision'


# paths
path = DataLoader.DATA_ROOT / clf_to_explain


class WordMetrics:
    def __init__(self, word):
        self.word = word
        self.cnt = 0
        self.cnt_pos = 0
        self.cnt_neg = 0
        self.sum = 0
        self.sum_pos = 0
        self.sum_neg = 0
        self.mean = 0
        self.mean_pos = 0
        self.mean_neg = 0

    def add(self, importance):
        self.sum += importance
        self.cnt += 1
        self.mean = self.sum / self.cnt
        if importance > 0:
            self.sum_pos += importance
            self.cnt_pos += 1
            self.mean_pos = self.sum_pos / self.cnt_pos
        elif importance < 0:
            self.sum_neg += importance
            self.cnt_neg += 1
            self.mean_neg = self.sum_neg / self.cnt_neg

    def to_df(self):
        return pd.DataFrame(data=[[self.word, self.mean, self.cnt, self.cnt_pos, self.cnt_neg, self.mean_pos, self.mean_neg]],
                            columns=cols)


class LSTMMetrics:
    def __init__(self, word):
        self.word = word
        self.cnt = 0
        self.sum = 0
        self.mean = 0
        self.cnt_pos = None
        self.cnt_neg = None
        self.sum_pos = None
        self.sum_neg = None
        self.mean_pos = None
        self.mean_neg = None

    def add(self, importance):
        self.sum += importance
        self.cnt += 1
        self.mean = self.sum / self.cnt

    def to_df(self):
        return pd.DataFrame(data=[[self.word, self.mean, self.cnt, self.cnt_pos, self.cnt_neg, self.mean_pos, self.mean_neg]],
                            columns=cols)

def combine_explanation_in_matrix(binary=False):
    explanation_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.p') and 'explanation' in file]
    explanation_files.sort(key=natural_sort_key)
    explanations = []
    word_to_idx = {}
    idx_to_word = []
    idx = 0
    # create word_to_idx and idx_to_word
    for f in explanation_files:
        exp = pickle.load(open(f, 'rb'))
        exp = exp.as_list()
        explanations.append(exp)
        for word, _ in exp:
            word = lemmatizer.lemmatize(word.lower())
            if word not in word_to_idx:
                idx_to_word.append(word)
                word_to_idx[word] = idx
                idx += 1
    
    words_idx_tuple = sorted(word_to_idx.items(), key=lambda x: x[1]) # sort words by idx in increasing order
    words = [item[0] for item in words_idx_tuple] # get only the words

    samples = len(explanations)
    features = len(word_to_idx)
    exp_bow = np.zeros((samples, features))
    for i, exp in enumerate(explanations):
        for word, importance in exp:
            word = lemmatizer.lemmatize(word.lower())
            if binary:
                exp_bow[i][word_to_idx[word]] = 1
            else:
                exp_bow[i][word_to_idx[word]] = importance

    return exp_bow, words


def get_explanation_labels():
    explanation_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.txt') and 'explanation' in file]
    explanation_files.sort(key=natural_sort_key)
    predicted_labels = []
    actual_labels = []
    for f in explanation_files:
        f = open(f, 'r')
        lines = [line for line in f.readlines()]
        f.close()
        prediction = float(re.findall("\d+\.\d+", lines[1])[0]) # read the predicted value (output of the model)
        if prediction >= 0.5:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)
        if 'accepted' in lines[3]:
            actual_labels.append(1)
        elif 'rejected' in lines[3]:
            actual_labels.append(0)
    return np.array(predicted_labels), np.array(actual_labels)
    

def combine_explanations(clf_to_explain=clf_to_explain, criterion='abs'):
    """
    :param criterion: 'abs' | 'pos' | 'neg'
    """
    path = DataLoader.DATA_ROOT / clf_to_explain
    explanation_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.p') and 'explanation' in file]
    explanation_files.sort(key=natural_sort_key)

    # combine all explanations
    word_metrics_acc = {}

    for idx, f in enumerate(explanation_files):
        exp = pickle.load(open(f, 'rb'))
        exp_list = exp.as_list()
        for word, importance in exp_list:
            if word.lower() in word_metrics_acc:
                word_metrics_acc[word.lower()].add(importance)
            else:
                word_metrics_acc[word.lower()] = WordMetrics(word)
                word_metrics_acc[word.lower()].add(importance)

    # to dataframe for printing
    combined_words = pd.DataFrame(columns=cols)

    for metric in word_metrics_acc.values():
        combined_words = combined_words.append(metric.to_df(),  ignore_index=True)

    if criterion == 'abs':
        # combined_words = combined_words.sort_values(['#Total', 'Mean'], ascending=[False, False])
        abs_df = combined_words.copy()
        abs_df['Mean'] = abs_df['Mean'].abs()
        combined_words = combined_words.reindex(abs_df.sort_values('Mean', ascending=False).index)
    elif criterion == 'pos':
        pos_df = combined_words.copy()
        combined_words = combined_words.reindex(pos_df.sort_values('Mean', ascending=False).index)
    elif criterion == 'neg':
        # combined_words = combined_words.sort_values(['#Total', 'Mean'], ascending=[False, False])
        neg_df = combined_words.copy()
        combined_words = combined_words.reindex(neg_df.sort_values('Mean', ascending=True).index)
    
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(combined_words)
    return combined_words
