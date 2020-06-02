import pickle
import pandas as pd
from DataLoader import DataLoader


cols = ['Words', 'Mean', '#Total', '#Positive', '#Neg', 'PosMean', 'NegMean']
final_decision = 'only'
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


indexes = range(0, 50)

# combine all explanations
word_metrics_acc = {}

for idx in indexes:
    exp = pickle.load(open(path / ('explanation' + str(idx) + '.p'), 'rb'))
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


# combined_words = combined_words.sort_values(['#Total', 'Mean'], ascending=[False, False])
abs_df = combined_words.copy()
abs_df['Mean'] = abs_df['Mean'].abs()
combined_words = combined_words.reindex(abs_df.sort_values('Mean', ascending=False).index)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(combined_words)
