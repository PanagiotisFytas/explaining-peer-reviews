import re
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report


NaturalSortRegex = re.compile('([0-9]+)')


# for sorting with natural ordering
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(NaturalSortRegex, s)]


def rmse(predictions, targets):
    """
    RMSE loss for numpy
    :param predictions: Numpy array of predictions
    :param targets: Numpy array of targets
    :return: RMSE loss
    """
    return torch.sqrt(((predictions - targets) ** 2).mean())


class Metrics:
    def __init__(self, data=None, model=None):
        if not data or not model:
            self.confusion_matrix = None
            self.precision = 0
            self.recall = 0
            self.f1 = 0
            self.report = {}
        else:
            model.eval()
            embeddings, lengths, labels = data
            predictions = model(embeddings, lengths)
            preds = (predictions.view(-1) >= 0.5).to(device='cpu', dtype=torch.int)
            targets = (labels >= 0.5).to(device='cpu', dtype=torch.int)
            self.confusion_matrix = confusion_matrix(targets.numpy(), preds.numpy())
            self.precision = precision_score(targets.numpy(), preds.numpy())
            self.recall = recall_score(targets, preds)
            self.f1 = f1_score(targets, preds)
            self.report = classification_report(targets, preds, output_dict=True)

    def __add__(self, other):
        result = Metrics()
        result.confusion_matrix = self.confusion_matrix + other.confusion_matrix
        result.precision = self.precision + other.precision
        result.recall = self.recall + other.recall
        result.f1 = self.f1 + other.f1
        for label, scores in self.report.items():
            if isinstance(scores, dict):
                result.report[label] = {}
                for metric, value in scores.items():
                    result.report[label][metric] = self.report[label][metric] + other.report[label][metric]
            else:
                result.report[label] = self.report[label] + other.report[label]
        return result

    def __truediv__(self, other):
        self.confusion_matrix = self.confusion_matrix / other
        self.precision = self.precision / other
        self.recall = self.recall / other
        self.f1 = self.f1 / other
        for label, scores in self.report.items():
            if isinstance(scores, dict):
                for metric, value in scores.items():
                    self.report[label][metric] = self.report[label][metric] / other
            else:
                self.report[label] = self.report[label] / other
        return self

    def __str__(self):
        # txt = '''
        # Precision: {0:.2f}
        # Recall: {1:.2f}
        # F1: {1:.2f}
        # '''.format(self.precision, self.recall, self.f1) + self.report.__str__()
        txt = '''
        Accuracy:    {0:.2f}
        Macro F1:    {1:.2f}
        Weighted F1: {2:.2f}
        '''.format(self.report['accuracy'], self.report['macro avg']['f1-score'],
                   self.report['weighted avg']['f1-score'])
        return txt

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def __rtruediv__(self, other):
        if other == 1:
            return self
        else:
            return self.__add__(other)


def cross_validation_metrics(network, network_params, optimizer_class, loss_fn_class, lr, epochs, batch_size, device, data, k=5,
                     shuffle=True):
    input, lengths, labels = data
    training_parameters = {'epochs': epochs, 'batch_size': batch_size}

    num_samples = input.shape[0]
    indices = list(range(num_samples))

    if shuffle:
        np.random.shuffle(indices)

    cv_metrics = []
    start = 0
    end = int(np.floor(1 / k * num_samples))
    samples_in_fold = end - start
    for i in range(k):
        print('Fold: ', i+1, ' Start: ', start, ' End: ', end)
        model = network(**network_params)
        model.to(device)
        optimizer = optimizer_class(model.parameters(), lr=lr)
        loss_fn = loss_fn_class()

        valid_idx = indices[start:end]
        train_idx = np.append(indices[:start], indices[end:])

        valid_input = input[valid_idx, :, :]
        valid_length = lengths[valid_idx]
        valid_labels = labels[valid_idx]

        train_input = input[train_idx, :, :]
        train_length = lengths[train_idx]
        train_labels = labels[train_idx]

        train_data = [train_input, train_length, train_labels]
        valid_data = [valid_input, valid_length, valid_labels]
        training_loop(train_data, valid_data, model, device, optimizer, loss_fn, **training_parameters, verbose=False)
        metrics = Metrics(valid_data, model)
        print(metrics)
        cv_metrics.append(metrics)

        start = end
        end = min(end + samples_in_fold, num_samples)

    print(sum(cv_metrics))
    print(sum(cv_metrics)/k)


def training_loop(data, test_data, model, device, optimizer, loss_fn, epochs=100, batch_size=64, verbose=True):
    embeddings, lengths, labels = data
    test_embeddings, test_lengths, test_labels = test_data
    N, _seq_len, input_size = embeddings.shape
    test_N, _, _ = test_embeddings.shape
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(N)
        model.train()
        for i in range(0, N, batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_y = labels[indices]
            batch_x = embeddings[indices, :, :]
            # for data augmentation
            # _, seq_len, _ = batch_x.shape
            # seq_permutation = torch.randperm(seq_len)
            # batch_x = batch_x[:, seq_permutation, :]
            batch_x.to(device)
            batch_y.to(device)

            batch_lengths = lengths[indices]
            batch_lengths.to(device)

            preds = model(batch_x, batch_lengths).squeeze(1)
            loss = loss_fn(preds, batch_y)
            train_loss = loss.item()
            loss.backward()
            model.hx = model.hx.detach()
            optimizer.step()

        if verbose:
            model.eval()

            predictions = model(embeddings, lengths)
            preds = predictions.view(-1) >= 0.5
            targets = labels >= 0.5

            accuracy = (preds == targets).sum() * (1 / N)
            print('-----EPOCH ' + str(epoch) + '-----')
            print('Accuracy on train set: ', accuracy)

            predictions = model(test_embeddings, test_lengths)
            preds = (predictions.view(-1) >= 0.5).to(device='cpu', dtype=torch.int)
            targets = (test_labels >= 0.5).to(device='cpu', dtype=torch.int)

            # print(preds.shape)
            # print(targets.shape)
            accuracy = (preds == targets).sum() * (1 / test_N)
            print('Accuracy on test set: ', accuracy)
            print('Confusion on test set: ', confusion_matrix(targets.numpy(), preds.numpy()))
            print('Precision on test set: ', precision_score(targets.numpy(), preds.numpy()))
            print('Recall on test set: ', recall_score(targets, preds))
            print('F1 on test set: ', f1_score(targets, preds))
            print('Report:\n', classification_report(targets, preds))
            print('-----------------')
            # predictions = model(test_data).squeeze(1)
            # print('RMSE on test set: ', rmse(predictions, test_labels))




#
# class CustomDataset(torch.utils.data.Dataset):
#     def __init__(self, input, lengths, Y):
#         self.input = input
#         self.lengths = lengths
#         self.Y = Y
#         self.len = len(input)
#         self.cnt = 1
#
#     def __getitem__(self, idx):
#         print(self.cnt)
#         self.cnt += 1
#         # print({'inp': self.input[idx], 'lengths': self.lengths[idx].item()}, self.Y[idx])
#         if isinstance(idx, Iterable):
#             return {'inp': self.input[idx], 'lengths': self.lengths[idx]}, self.Y[idx]
#         else:
#             return {'inp': self.input[idx], 'lengths': self.lengths[idx].item()}, self.Y[idx]
#         # if self.cnt >= self.len:
#         #     return None
#         # X, y = {'inp': self.input[idx, :, :], 'lengths': self.lengths[idx]}, self.Y[idx]
#         # y = y.item() if y is None else y
#         # Xinp = X['inp'].item() if X['inp'] is None else X['inp']
#         # Xlengths = X['lengths'].item() if X['lengths'] is None else X['lengths']
#         # X, y = {'inp': Xinp, 'lengths': Xlengths}, y
#         # return X, y
#
#     def __len__(self):
#         print(self.len)
#         return self.len

