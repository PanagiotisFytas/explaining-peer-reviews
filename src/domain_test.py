from DataLoader import DataLoader
import numpy as np
import scipy.stats as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import cross_val_predict


data_loader = DataLoader(device='cpu')

domains = data_loader.read_email_domains()
# print(domains)
labels = data_loader.read_labels().numpy()

unis = ['google.com', 'fb.com', 'microsoft.com', 'cs.cmu.edu', 'openai.com', 'stanford.edu', 'berkeley.edu', 'umontreal.ca', 'us.ibm.com']
print(data_loader.domain_counts(domains))

count = 0
has_prestigious_author = np.zeros(len(domains))
for idx, paper_domains in enumerate(domains):
    for pr in unis:
        if pr in paper_domains:
            count+=1
            has_prestigious_author[idx] = 1
            break
has_prestigious_author.reshape(-1,1)
print(count)

valid_size = 0.1

num_train = len(labels)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))


train_idx, test_idx = indices[split:], indices[:split]

test_labels = labels[test_idx]
train_labels = labels[train_idx]

test_has_prestigious_authors = has_prestigious_author[test_idx]

train_has_prestigious_authors = has_prestigious_author[train_idx]

clf =  LogisticRegression(max_iter=500).fit(train_has_prestigious_authors, train_labels)
print(clf.score(test_has_prestigious_authors, test_labels))
preds_prob = clf.predict_proba(test_has_prestigious_authors)[:, 0]
print('MSE with probs', mean_squared_error(test_labels, preds_prob))
preds = clf.predict(test_has_prestigious_authors)
print('MSE with labels', mean_squared_error(test_labels, preds))
print('Classification report:\n', classification_report(test_labels, preds))
print('#############################################')
