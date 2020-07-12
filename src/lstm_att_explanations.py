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


device_idx = input("GPU: ")
GPU = True
if GPU:
    device = torch.device("cuda:" + device_idx if torch.cuda.is_available() else "cpu")
else:
    device = torch.device("cpu")
print(device)


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
                


if __name__ == '__main__':

    clf_to_explain = 'lstm_att_classifier'
    # paths
    path = LSTMEmbeddingLoader.DATA_ROOT / clf_to_explain
    model_path = str(path / 'model.pt')


    data_loader = LSTMEmbeddingLoader(device=device,
                                    lemmatise=True, 
                                    lowercase=True, 
                                    stopword_removal=False, 
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

    # load model
    model = torch.load(model_path)
    model.to(device)

    generate_lstm_explanations(model, test_text_input, test_embeddings_input, test_number_of_tokens)