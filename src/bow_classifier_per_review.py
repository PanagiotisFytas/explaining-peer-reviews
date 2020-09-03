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
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, classification_report
from helper_functions import training_loop, cross_validation_metrics
from models import BoWClassifier


try:
    stopwords_en = stopwords.words('english')
except LookupError:
    import nltk
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


def generate_freq_bow_for_lexicon(lexicon, word_to_idx, reviews, grams=(2,3,4), freq=True):
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
        # print(idx)
        bow_vector = np.zeros(L)
        n_grams = {}
        for n in grams:
            n_grams[n] = list(ngrams(review, n))
        N = len(review)
        for word in review:
            if word in word_to_idx:
                if freq:
                    bow_vector[word_to_idx[word]] = review.count(word) /  N
                else:
                    bow_vector[word_to_idx[word]] = float(word in review)
        for n in grams:
            for n_gram in n_grams[n]:
                if n_gram in word_to_idx:
                    if freq:
                        bow_vector[word_to_idx[n_gram]] = n_grams[n].count(n_gram) / (N-n+1)
                    else:
                        bow_vector[word_to_idx[n_gram]] = float(n_gram in n_grams[n])

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

if __name__ == "__main__":
    with open('src/config/bow_classifier_per_review.yaml') as f:
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
    causal_lexicon = config['causal_lexicon']

    lexicon_size = config['lexicon_size']
    if causal_lexicon:
        clf_to_explain = 'lstm_att_classifier_per_review' + causal_lexicon
    else:
        clf_to_explain = 'lstm_att_classifier_per_review'
    path = LSTMPerReviewDataLoader.DATA_ROOT / clf_to_explain

    ################################################
    ########### Read Filtered Lexicon ##############
    ################################################


    try:
        from ast import literal_eval
        with open(path / 'filtered_lexicon.txt') as f:
            filtered_lexicon = [list(literal_eval(line)) for line in f][0]
    except FileNotFoundError:

        lexicon_path = str(path / 'lexicon.csv')
        lexicon = pd.read_csv(lexicon_path)
        lexicon = lexicon.head(lexicon_size)['Words'].tolist()

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

        f = open(path / 'filtered_lexicon.txt', "w")
        f.write(str(filter_lexicon))
        f.close()
        

    ################################################
    ################################################
    ################################################

    lexicon_size = len(filtered_lexicon)
    print('Filtered Lexicon Size: ', lexicon_size)

    word_to_idx = get_word_to_idx(filtered_lexicon)

    bow = generate_freq_bow_for_lexicon(filtered_lexicon, word_to_idx, text_input, freq=False).to_numpy()
    bow = torch.tensor(bow).float()



    # ## SKLEARN Logistic Regression ###
    # train_bow = generate_freq_bow_for_lexicon(filtered_lexicon, word_to_idx, train_text_input, freq=False)
    # test_bow = generate_freq_bow_for_lexicon(filtered_lexicon, word_to_idx, test_text_input, freq=False)


    # train_labels = train_labels.numpy()
    # test_labels = test_labels.numpy()


    # clf =  LogisticRegression(max_iter=100000).fit(train_bow, train_labels)
    # print(clf.score(test_bow, test_labels))
    # preds_prob = clf.predict_proba(test_bow)[:, 0]
    # print('MSE with probs', mean_squared_error(test_labels, preds_prob))
    # preds = clf.predict(test_bow)
    # print('MSE with labels', mean_squared_error(test_labels, preds))
    # print('Classification report:\n', classification_report(test_labels, preds))

    # exit()
    # #####

    if causal_layer == 'residual':
        nn_conf = config[causal_layer]
    else:
        nn_conf = config['not_residual']


    epochs = nn_conf['epochs'] # 200 # 500
    batch_size = nn_conf['batch_size'] # 300 # 120
    lr = nn_conf['lr'] # 0.0001
    hidden_dimensions = nn_conf['hidden_dimensions'] #[128, 64] # [1500, 700, 300] # [700, 300]
    causal_hidden_dimensions = nn_conf['causal_hidden_dimensions'] # [64]
    bow_hidden_dimensions = nn_conf['bow_hidden_dimensions'] # [64]
    dropout = nn_conf['dropout']
    dropout2 = nn_conf['dropout2']
    activation = nn_conf['activation']
    activation2 = nn_conf['activation2']


    if cross_validation:
        network = BoWClassifier
        network_params = {
            'input_size': lexicon_size,
            'hidden_dimensions': hidden_dimensions,
            'causal_hidden_dimensions': causal_hidden_dimensions,
            'bow_hidden_dimensions': bow_hidden_dimensions,
            'dropout': dropout,
            'dropout2': dropout2,
            'activation': activation,
            'activation2': activation2,
            'causal_layer': causal_layer
        }
        optimizer = torch.optim.Adam
        lr = lr
        loss_fn = nn.BCELoss
        if not causal_layer:
            data = [bow, labels, labels]
            cross_validation_metrics(network, network_params, optimizer, loss_fn, lr,
                                    epochs, batch_size, device, data, k=5, shuffle=True)
        else:
            data = [bow, labels, labels, confounders]
            confounding_loss_fn = nn.MSELoss
            cross_validation_metrics(network, network_params, optimizer, loss_fn, lr,
                                    epochs, batch_size, device, data, confounding_loss_fn=confounding_loss_fn, k=5, shuffle=True)


        cross_validation_metrics(network, network_params, optimizer, loss_fn, lr,
                                epochs, batch_size, device, data, k=5, shuffle=True)
        # # dataset = CustomDataset(embeddings_input, number_of_reviews, labels)
        # dataset = Dataset({'inp': embeddings_input, 'lengths': number_of_reviews}, labels)
        # # X_dict = {'inp': embeddings_input, 'lengths': number_of_reviews}
        # print(embeddings_input.shape, number_of_reviews.shape, labels.shape)
        # net.fit(dataset, y=labels)
        # preds = cross_val_predict(net, dataset, y=labels.to('cpu'), cv=5)
    else:
        # hold-one-out split
        model = BoWClassifier(device,
                            input_size=lexicon_size,
                            hidden_dimensions=hidden_dimensions,
                            causal_hidden_dimensions=causal_hidden_dimensions,
                            bow_hidden_dimensions=bow_hidden_dimensions,
                            dropout=dropout,
                            dropout2=dropout2,
                            activation=activation,
                            activation2=activation2,
                            causal_layer=causal_layer)
        shuffle = False
        valid_size = 0.1

        num_train = bow.shape[0]
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle:
            np.random.shuffle(indices)

        if config['on_validation_set']:
            # will test on validation set and train on train set
            train_idx, test_idx = indices[2*split:], indices[split:2*split]
        else:
            # will test on test set and train on train and vaildation set
            train_idx, test_idx = indices[split:], indices[:split]

        test_bow = bow[test_idx, :]
        test_labels = labels[test_idx]

        if causal_layer:
            test_confounders = confounders[test_idx, :]


        bow = bow[train_idx, :]
        labels = labels[train_idx]

        if causal_layer:
            confounders = confounders[train_idx, :]


        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()
        if not causal_layer:
            confounding_loss_fn = None
            data = [bow, labels, labels]
            test_data = [test_bow, test_labels, test_labels]
        else:
            confounding_loss_fn = nn.BCELoss()
            data = [bow, labels, labels, confounders]
            test_data = [test_bow, test_labels, test_labels, test_confounders]

        model.to(device)


        losses = training_loop(data,
                            test_data, 
                            model, 
                            device, 
                            optimizer, 
                            loss_fn,
                            confounder_loss_fn=confounding_loss_fn,
                            causal_layer=causal_layer,
                            epochs=epochs, 
                            batch_size=batch_size, 
                            return_losses=True,
                            loss2_mult=config['loss2_mult']
                            )


        if not causal_layer:
            train_losses, test_losses = losses
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.legend()
            plt.savefig('/home/pfytas/peer-review-classification/bow_losses.png')
            model_path = LSTMPerReviewDataLoader.DATA_ROOT / 'bow_classifier_per_review'
        else:
            train_losses, test_losses, confounding_train_losses, confounding_test_losses = losses
            plt.plot(train_losses, label='Train Loss')
            plt.plot(test_losses, label='Test Loss')
            plt.plot(confounding_train_losses, label='Confounding Train Loss')
            plt.plot(confounding_test_losses, label='Confounding Test Loss')
            plt.legend()
            # plt.yscale('log')
            plt.savefig('/home/pfytas/peer-review-classification/bow_losses.png')
            model_path = LSTMPerReviewDataLoader.DATA_ROOT / ('bow_classifier_per_review' + causal_layer)
        model_path.mkdir(parents=True, exist_ok=True)
        torch.save(model, model_path / 'model.pt')
