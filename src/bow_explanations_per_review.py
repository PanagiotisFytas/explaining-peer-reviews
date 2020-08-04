import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from DataLoader import LSTMPerReviewDataLoader
import numpy as np
from helper_functions import training_loop, cross_validation_metrics
from models import BoWClassifier
import pathlib
import os
from combine_explanations import BoWMetrics
import pandas as pd
import scipy.stats as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import cross_val_predict
import yaml
from nltk import ngrams
from bow_classifier_per_review import get_word_to_idx

cols = ['Words', 'Mean', '#Total', '#Positive', '#Neg', 'PosMean', 'NegMean']

def generate_bow_explanations(model, reviews, filtered_lexicon, complex_explanations=True, get_only_test_words=True, grams=(2,3,4)):
    '''
    generate global explanations
    The reviews must already be preprocessed. The lstm explanations generator will not do preprocessing
    '''
    model.eval()
    # predicted_labels = model(embeddings, number_of_tokens) # It may be useful to keep track of local explanations: labels and specific weights
    bow_weights = model.bow_fc_layers[0].weight
    print(bow_weights.shape)
    if complex_explanations:
        # e_weights = model.fc_layers[0].weight # TODO projection
        # importances = torch.matmul(model.last_fc.weight, torch.matmul(e_weights, bow_weights)).view(-1, 1)
        e_weights = model.last_fc.weight # TODO projection
        print(e_weights.shape)
        importances = torch.matmul(e_weights, bow_weights).view(-1, 1)
        
        print(importances.shape)
    else:
        importances = torch.norm(bow_weights, p=1, dim=0)
    word_metrics_acc = {}
    
    if not get_only_test_words:    
        # get the weight of all n_grams
        for i, n_gram in enumerate(filtered_lexicon):
            weight = importances[i].item()
            word_metrics_acc[n_gram] = BoWMetrics(n_gram, complex_explanations)
            word_metrics_acc[n_gram].add(weight)
    
    word_to_idx = get_word_to_idx(filtered_lexicon)

    for review in reviews:
        for n in grams:
            review += list(ngrams(review, n))                
        for n_gram in review:
            if n_gram in word_to_idx:
                if get_only_test_words and n_gram not in word_metrics_acc:
                    # get the weight of only the n_grams in the test set
                    weight = importances[word_to_idx[n_gram]].item()
                    word_metrics_acc[n_gram] = BoWMetrics(n_gram, complex_explanations)
                    word_metrics_acc[n_gram].add(weight)
                word_metrics_acc[n_gram].increment_count()
    # to dataframe for printing
    combined_words = pd.DataFrame(columns=cols)

    for metric in word_metrics_acc.values():
        combined_words = combined_words.append(metric.to_df(),  ignore_index=True)
    
    pos_df = combined_words.copy()
    combined_words = combined_words.reindex(pos_df.sort_values('Mean', ascending=False).index)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(combined_words.head(200))
    return combined_words
                

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
    for review in reviews:
        bow_vector = []
        for word in lexicon:
            if word in review:
                bow_vector.append(1)
            else:
                bow_vector.append(0)
        bow.loc[idx] = bow_vector
        idx += 1
    return bow
        


# code from https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
def cramers_V(x, y):
    '''
    :param x: panda df
    :param y: panda df
    :returns the cramer V correlation between x and y
    '''
    confusion_matrix = pd.crosstab(y, x)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))


if __name__ == '__main__':

    with open('src/config/bow_classifier_per_review.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    device_idx = config['CUDA']
    GPU = True
    if GPU:
        device = torch.device("cuda:" + device_idx if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(device)

    # causal_layer = None | 'adversarial' | 'residual'
    causal_layer = config['causal_layer']
    lexicon_size = config['lexicon_size']
    if not causal_layer:
        clf_to_explain = 'bow_classifier_per_review'
    else:
        clf_to_explain = 'bow_classifier_per_review' + causal_layer
    # paths
    path = LSTMPerReviewDataLoader.DATA_ROOT / clf_to_explain
    model_path = str(path / 'model.pt')

    lexicon_size = config['lexicon_size']
    causal_lexicon = config['causal_lexicon']

    if causal_lexicon:
        lexicon_clf = 'lstm_att_classifier_per_review' + causal_lexicon
    else:
        lexicon_clf = 'lstm_att_classifier_per_review'
    lexicon_path = LSTMPerReviewDataLoader.DATA_ROOT / lexicon_clf
    # read filtered lexicon that the bow uses
    from ast import literal_eval
    with open(lexicon_path / 'filtered_lexicon.txt') as f:
        filtered_lexicon = [list(literal_eval(line)) for line in f][0]

    data_loader = LSTMPerReviewDataLoader(device=device,
                                          lemmatise=True, 
                                          lowercase=True, 
                                          remove_stopwords=False, 
                                          punctuation_removal=True,
                                          final_decision='exclude',
                                          pretrained_weights='scibert_scivocab_uncased',
                                          )

    embeddings_input = data_loader.read_embeddigns_from_file()
    reviews = data_loader.read_reviews_only_text()

    number_of_tokens = torch.tensor([review.shape[0] for review in embeddings_input]).to(device)
    embeddings_input = rnn.pad_sequence(embeddings_input, batch_first=True)  # pad the reviews to form a tensor
    print(embeddings_input.shape)
    labels = data_loader.read_labels().to(device)
    _, _, embedding_dimension = embeddings_input.shape
    aspect = config['aspect']
    if aspect == 'abstract':
        confounders = data_loader.read_abstract_embeddings()
        confounders = data_loader.copy_to_peer_review(confounders)
        confounders = torch.tensor(confounders, dtype=torch.float)
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


    text_input = np.array(data_loader.preprocessor.preprocess(reviews))

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

    # load model
    model = torch.load(model_path)
    model.to(device)
    model.device = device

    test_embeddings_input = test_embeddings_input.to(device)
    exp = generate_bow_explanations(model, test_text_input, filtered_lexicon, get_only_test_words=True)
    exp['Mean'] = exp['Mean'].abs()
    # get explanation (and lexicon) from test set

    test_bow = generate_bow_for_lexicon(exp, test_text_input, k=lexicon_size)

    test_labels_df = pd.DataFrame(test_labels.view(-1,1).to('cpu').numpy())

    train_bow = generate_bow_for_lexicon(exp, train_text_input, k=lexicon_size)

    train_labels_df = pd.DataFrame(train_labels.to('cpu').numpy())


    # with pd.option_context('display.max_rows', None, 'display.max_columns', 8):
        # print(test_bow)
        # print(test_labels_df)
    
    if aspect != 'abstract' :
        print('###############CORRELATION###################')    
        correlations = []
        X = pd.concat([train_bow, test_bow])
        y = confounders
        _, number_of_features = y.shape
        for word_idx in range(lexicon_size):
            for feature in range(number_of_features):
                correlations.append(ss.pointbiserialr(X.iloc[:, word_idx], y[:, feature]))
        print("Average PointBiserial Correlation: ", np.mean(correlations))
        print('#############################################')

    print('###### LR on lexicon (no confounding): ######')
    if config['cv_explanation']:
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
    if config['cv_explanation']:
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
    if config['cv_explanation']:
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

    # print('Saving Lexicon')
    # lexicon_path = str(path / 'lexicon.csv')
    # exp.to_csv(lexicon_path, index=False)

            
