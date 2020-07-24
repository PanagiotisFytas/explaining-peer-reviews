import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from DataLoader import LSTMEmbeddingLoader
import numpy as np
from helper_functions import training_loop, cross_validation_metrics
from models import LSTMAttentionClassifier
import pathlib
import os
from combine_explanations import LSTMMetrics, cols
import pandas as pd
import scipy.stats as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.model_selection import cross_val_predict
import yaml


def generate_lstm_explanations(model, reviews, embeddings, number_of_tokens):
    '''
    generate global explanations
    The reviews must already be preprocessed. The lstm explanations generator will not do preprocessing
    '''
    model.eval()
    # predicted_labels = model(embeddings, number_of_tokens) # It may be useful to keep track of local explanations: labels and specific weights
    attention_weights, _ = model.rnn_att_forward(embeddings, number_of_tokens)
    print(attention_weights.shape)
    word_metrics_acc = {}
    for i, review in enumerate(reviews):
        # print('Review: \n', review)
        for j, word in enumerate(review):
            weight = attention_weights[i, j].item()
            # print('Word: ', word, ' Weight: ', weight, '\n')
            if word not in word_metrics_acc:    
                word_metrics_acc[word] = LSTMMetrics(word)
            word_metrics_acc[word].add(weight)

        # to dataframe for printing
    combined_words = pd.DataFrame(columns=cols)

    for metric in word_metrics_acc.values():
        combined_words = combined_words.append(metric.to_df(),  ignore_index=True)
    
    pos_df = combined_words.copy()
    combined_words = combined_words.reindex(pos_df.sort_values('Mean', ascending=False).index)

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(combined_words)
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

    with open('src/config/lstm_att_classifier.yaml') as f:
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
        clf_to_explain = 'lstm_att_classifier'
    else:
        clf_to_explain = 'lstm_att_classifier' + causal_layer
    # paths
    path = LSTMEmbeddingLoader.DATA_ROOT / clf_to_explain
    model_path = str(path / 'model.pt')


    data_loader = LSTMEmbeddingLoader(device=device,
                                    lemmatise=True, 
                                    lowercase=True, 
                                    remove_stopwords=False, 
                                    punctuation_removal=True,
                                    final_decision='only',
                                    pretrained_weights='scibert_scivocab_uncased',
                                    )

    embeddings_input = data_loader.read_embeddigns_from_file()


    number_of_tokens = torch.tensor([review.shape[0] for review in embeddings_input]).to(device)
    embeddings_input = rnn.pad_sequence(embeddings_input, batch_first=True).to(device)  # pad the reviews to form a tensor
    print(embeddings_input.shape)
    labels = data_loader.read_labels().to(device)
    _, _, embedding_dimension = embeddings_input.shape
    if causal_layer == 'residual':
        confounders = data_loader.read_abstract_embeddings()
    elif causal_layer == 'adversarial':
        confounders = data_loader.read_average_scores(aspect=config['aspect'])

    reviews = data_loader.read_reviews_only_text()
    text_input = np.array(data_loader.preprocessor.preprocess(reviews))

    valid_size = 0.1

    num_train = len(text_input)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))


    train_idx, test_idx = indices[split:], indices[:split]

    # test set

    test_text_input = text_input[test_idx]
    test_embeddings_input = embeddings_input[test_idx, :, :]
    test_number_of_tokens = number_of_tokens[test_idx]
    test_labels = labels[test_idx]
    if causal_layer:
        test_confounders = confounders[test_idx]

    # train set

    train_text_input = text_input[train_idx]
    train_embeddings_input = embeddings_input[train_idx, :, :]
    train_number_of_tokens = number_of_tokens[train_idx]
    train_labels = labels[train_idx]
    if causal_layer:
        train_confounders = confounders[train_idx]

    # load model
    model = torch.load(model_path)
    model.to(device)
    model.device = device

    exp = generate_lstm_explanations(model, test_text_input, test_embeddings_input, test_number_of_tokens)
    # get explanation (and lexicon) from test set

    test_bow = generate_bow_for_lexicon(exp, test_text_input, k=lexicon_size)

    test_labels_df = pd.DataFrame(test_labels.view(-1,1).to('cpu').numpy())

    train_bow = generate_bow_for_lexicon(exp, train_text_input, k=lexicon_size)

    train_labels_df = pd.DataFrame(train_labels.to('cpu').numpy())


    # with pd.option_context('display.max_rows', None, 'display.max_columns', 8):
        # print(test_bow)
        # print(test_labels_df)
    
    if causal_layer == 'adversarial':
        correlations = []
        X = pd.concat([train_bow, test_bow])
        y = pd.concat([train_labels_df, test_labels_df])
        for word_idx in range(lexicon_size):
            print(X.iloc[:, word_idx])
            print(y.reset_index(drop=True, inplace=False))
            correlations.append(ss.pointbiserialr(X.iloc[:, word_idx], y.reset_index(drop=True, inplace=False)))
        print("Average Correlation: ", np.mean(correlations))


    print('###### LR on lexicon (no confounding): ######')
    if config['cv_explanation']:
        X = pd.concat([train_bow, test_bow])
        y = pd.concat([train_labels_df, test_labels_df])
        preds = cross_val_predict(LogisticRegression(max_iter=500), X, y, cv=config['folds'])
        print('MSE with labels', mean_squared_error(y, preds))
        print('Classification report:\n', classification_report(y, preds))
    else:
        clf =  LogisticRegression(max_iter=500).fit(train_bow, train_labels_df)
        print(clf.score(test_bow, test_labels_df))
        preds_prob = clf.predict_proba(test_bow)[:, 0]
        print('MSE with probs', mean_squared_error(test_labels_df, preds_prob))
        preds = clf.predict(test_bow)
        print('MSE with labels', mean_squared_error(test_labels_df, preds))
        print('Classification report:\n', classification_report(test_labels_df, preds))
    print('#############################################')
    if causal_layer:
        print('###### LR on lexicon (confounders conf.): ######')

        # concatenate lexicon bag of words with confounders embeddins
        train_bow = train_bow.to_numpy()
        test_bow = test_bow.to_numpy()
        train_confounders = train_confounders.to('cpu').numpy()
        test_confounders = test_confounders.to('cpu').numpy()
        if causal_layer == 'adversarial':
            train_confounders = np.expand_dims(train_confounders, axis=1)
            test_confounders = np.expand_dims(test_confounders, axis=1)
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
            clf =  LogisticRegression(max_iter=500).fit(train_bow, train_labels_df)
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
            clf =  LogisticRegression(max_iter=500).fit(train_confounders, train_labels_df)
            print(clf.score(test_confounders, test_labels_df))
            preds_prob = clf.predict_proba(test_confounders)[:, 0]
            print('MSE with probs', mean_squared_error(test_labels_df, preds_prob))
            preds = clf.predict(test_confounders)
            print('MSE with labels', mean_squared_error(test_labels_df, preds))
            print('Classification report:\n', classification_report(test_labels_df, preds))
            print('#############################################')

            
