from combine_explanations import combine_explanations
import numpy as np
from DataLoader import DataLoader, PerReviewDataLoader
import torch

###########################
###### Read Reviews #######
###########################

# cuda
device_idx = input("GPU: ")

if device_idx not in [0, 1, 2, 3]:
    GPU = False
else:
    GPU = True

if GPU:
    device = torch.device("cuda:" + device_idx if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


clf_to_explain = 'final_decision_only'
# clf_to_explain = 'per_review_classifier'
if clf_to_explain == 'final_decision_only':
    final_decision = 'only'
    data_loader = DataLoader(device=device,
                             final_decision=final_decision,
                             allow_empty=False,
                             truncate_policy='right',
                             pretrained_weights='scibert_scivocab_uncased',
                             remove_duplicates=True,
                             remove_stopwords=False)
else:
    final_decision = 'exclude'
    data_loader = PerReviewDataLoader(device=device,
                                      final_decision=final_decision,
                                      allow_empty=False,
                                      truncate_policy='right',
                                      pretrained_weights='scibert_scivocab_uncased',
                                      remove_duplicates=True,
                                      remove_stopwords=False)



# Load Data

# data_loader.model.to(device)

text_input = np.array(data_loader.read_reviews_only_text())
if clf_to_explain == 'final_decision_only':
    labels = data_loader.read_labels().numpy()
else:
    labels = (data_loader.read_scores() > 5).to(dtype=torch.float).numpy()

valid_size = 0.1

num_train = len(text_input)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

train_idx, test_idx = indices[split:], indices[:split]

# test set

test_text_input = text_input[test_idx]
test_labels = labels[test_idx]

###########################

###########################
## unadjusted estimation ##
###########################

# check only the test labels
# text_input = test_text_input
# labels = test_labels

print(len(text_input))


salient_words = combine_explanations(clf_to_explain=clf_to_explain, criterion='pos')

# k = 15
for k in range(1, 25):
    most_important_words = salient_words.head(k)['Words'].to_numpy()

    print(most_important_words)

    words_exist = []

    for review in text_input:
        exist = 0
        if clf_to_explain == 'final_decision_only':
            lower_review = review[0].lower()
        else:
            lower_review = review.lower()
        for word in most_important_words:
            if word.lower() in lower_review:
                exist = 1
                break
        words_exist.append(exist)

    treatment = np.array(words_exist)
    # print(treatment)
    # print(len(treatment))
    # print(labels)
    # print(len(labels))
    print(treatment.mean())
    print(np.absolute(labels-treatment).mean())

    accepted_papers_with_treatment = 0
    accepted_papers_without_treatment = 0
    papers_with_treatment = 0
    papers_without_treatment = 0

    for idx, t in enumerate(treatment):
        if t == 0:
            papers_without_treatment +=1
            if labels[idx] == 1:
                accepted_papers_without_treatment += 1
        else:
            papers_with_treatment +=1
            if labels[idx] == 1:
                accepted_papers_with_treatment += 1

    unadjusted_ATE = accepted_papers_with_treatment/papers_with_treatment - accepted_papers_without_treatment/papers_without_treatment

    print('k: ', k)
    print('Unadjusted ATE: ', unadjusted_ATE)
    print()
###########################
