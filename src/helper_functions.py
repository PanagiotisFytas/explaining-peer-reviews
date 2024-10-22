import re
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, mean_squared_error
from math import sqrt


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
            if len(data) == 3:
                embeddings, lengths, labels = data
                predictions = model(embeddings, lengths)
            else:
                embeddings, lengths, labels, conf = data
                predictions, _ = model(embeddings, lengths, abstract=conf)
            # predictions = model(embeddings, lengths)
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
        Accuracy:    {0:.4f}
        Macro F1:    {1:.4f}
        Weighted F1: {2:.4f}
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


def cross_validation_metrics(network, network_params, optimizer_class, loss_fn_class, lr, epochs, batch_size, device,
                             data, causal_layer=None, k=5, gru_model=False, shuffle=True, loss2_mult=1,
                             confounding_loss_fn=None):
    """
    :param netwrok: the PyTorch model class
    :param network_params: the arguments of the constructor of the model
    :param optimzer_class: the PyTorch class to used for optimisation
    :param loss_fn_class: the class of the loss function
    :param lr: the learning rate
    :param epochs: the training epochs
    :param batch_size: the batch size of SGD
    :param device: 'cpu' | 'CUDA:0' | ...
    :param data: the data of the model
    :returns the cross validation metrics
    """
    if causal_layer:
        input, lengths, labels, conf = data
    else:
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
        if causal_layer:
            conf_loss_fn = confounding_loss_fn()
        else:
            conf_loss_fn = None

        valid_idx = indices[start:end]
        train_idx = np.append(indices[:start], indices[end:])

        # valid_input = input[valid_idx, :, :]
        valid_input = input[valid_idx]
        valid_length = lengths[valid_idx]
        valid_labels = labels[valid_idx]
        if causal_layer:
            valid_conf = conf[valid_idx]

        # train_input = input[train_idx, :, :]
        train_input = input[train_idx]
        train_length = lengths[train_idx]
        train_labels = labels[train_idx]
        if causal_layer:
            train_conf = conf[train_idx]


        train_data = [train_input.to(device), train_length.to(device), train_labels.to(device)]
        valid_data = [valid_input.to(device), valid_length.to(device), valid_labels.to(device)]
        if causal_layer:
            train_data.append(train_conf.to(device))
            valid_data.append(valid_conf.to(device))

        training_loop(train_data, valid_data, model, device, optimizer, loss_fn, **training_parameters,
                      gru_model=gru_model, verbose=False, loss2_mult=loss2_mult, confounder_loss_fn=conf_loss_fn,
                      causal_layer=causal_layer)
        metrics = Metrics(valid_data, model)
        print(metrics)
        cv_metrics.append(metrics)

        start = end
        end = min(end + samples_in_fold, num_samples)

    print(sum(cv_metrics))
    print(sum(cv_metrics)/k)


def training_loop(data, test_data, model, device, optimizer, loss_fn, confounder_loss_fn=None, epochs=100, batch_size=64, gru_model=False,
                  verbose=True, return_losses=False, causal_layer=None, task='classification', loss2_mult=1):
    '''
    :param causal_layer = None | 'adversarial'
    '''
    if not causal_layer:
        embeddings, lengths, labels = data
        test_embeddings, test_lengths, test_labels = test_data
    else:
        embeddings, lengths, labels, confounders = data
        test_embeddings, test_lengths, test_labels, test_confounders = test_data
        test_confounders = test_confounders.to(device)

    N = embeddings.shape[0]
    test_N = test_embeddings.shape[0]
    test_embeddings = test_embeddings.to(device)
    test_lengths = test_lengths.to(device)
    test_labels = test_labels.to(device)
    if return_losses:
        train_losses = []
        test_losses = []
        if causal_layer:
            confounder_train_losses = []
            confounder_test_losses = []
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(N)
        model.train()
        if return_losses:
            train_losses_in_epoch = []
            if causal_layer:
                confounder_train_losses_in_epoch = []
        for i in range(0, N, batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_y = labels[indices]
            # print('Emb ', embeddings.shape)
            if len(embeddings.shape) == 3:
                batch_x = embeddings[indices, :, :]
            else:
                batch_x = embeddings[indices, :]
            # print('BX ', batch_x.shape)
            if causal_layer:
                # print('C ', confounders.shape)
                if causal_layer == 'adversarial':
                    batch_confounders = confounders[indices]
                else:   
                    batch_confounders = confounders[indices, :]
                # print('BC ', batch_confounders.shape)
                batch_confounders = batch_confounders.to(device)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            batch_lengths = lengths[indices]
            batch_lengths = batch_lengths.to(device)

            if not causal_layer:
                preds = model(batch_x, batch_lengths).squeeze(1)
                loss = loss_fn(preds, batch_y)
                train_loss = loss.item()

                loss.backward()

                if return_losses:
                    train_losses_in_epoch.append(train_loss)
            elif causal_layer == 'adversarial':
                preds, confounder_preds = model(batch_x, batch_lengths)
                preds = preds.squeeze(1)
                confounder_preds = confounder_preds.squeeze(1)
                # paper classification loss
                loss1 = loss_fn(preds, batch_y)
                train_loss = loss1.item()
                # confounder loss
                loss2 = confounder_loss_fn(confounder_preds, batch_confounders)
                confounder_train_loss = loss2.item()
                # total loss
                # loss = loss1 - .1 * loss2
                loss = loss1 + loss2_mult * loss2
                loss.backward()
                
                if return_losses:
                    train_losses_in_epoch.append(train_loss)
                    confounder_train_losses_in_epoch.append(confounder_train_loss)
                
            elif causal_layer == 'residual':
                preds, confounder_preds = model(batch_x, batch_lengths, abstract=batch_confounders)
                preds = preds.squeeze(1)
                confounder_preds = confounder_preds.squeeze(1)
                # paper classification loss
                loss1 = loss_fn(preds, batch_y)
                train_loss = loss1.item()
                # confounder loss
                loss2 = confounder_loss_fn(confounder_preds, batch_y)
                confounder_train_loss = loss2.item()
                # total loss
                loss = loss1 + loss2_mult*loss2

                loss.backward()
                
                if return_losses:
                    train_losses_in_epoch.append(train_loss)
                    confounder_train_losses_in_epoch.append(confounder_train_loss)
                

            if gru_model:
                model.hx = model.hx.detach()
            optimizer.step()

        if return_losses:
            model.eval()
            if not causal_layer:
                test_preds = model(test_embeddings, test_lengths).squeeze(1)
            elif causal_layer == 'adversarial':
                test_preds, test_confounder_preds = model(test_embeddings, test_lengths)
                test_preds = test_preds.squeeze(1)
                test_confounder_preds = test_confounder_preds.squeeze(1)
            elif causal_layer == 'residual':
                test_preds, test_confounder_preds = model(test_embeddings, test_lengths, abstract=test_confounders)
                test_preds = test_preds.squeeze(1)
                test_confounder_preds = test_confounder_preds.squeeze(1)
            test_loss = loss_fn(test_preds, test_labels)
            test_loss = test_loss.item()
            train_loss = np.array(train_losses_in_epoch).mean()
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            if causal_layer == 'adversarial':
                confounder_test_loss = confounder_loss_fn(test_confounder_preds, test_confounders)
                confounder_test_loss = confounder_test_loss.item()
                confounder_train_loss = np.array(confounder_train_losses_in_epoch).mean()
                confounder_train_losses.append(confounder_train_loss)
                confounder_test_losses.append(confounder_test_loss)
            elif causal_layer == 'residual':
                confounder_test_loss = confounder_loss_fn(test_confounder_preds, test_labels)
                confounder_test_loss = confounder_test_loss.item()
                confounder_train_loss = np.array(confounder_train_losses_in_epoch).mean()
                confounder_train_losses.append(confounder_train_loss)
                confounder_test_losses.append(confounder_test_loss)



        if verbose:
            model.eval()
            print('-----EPOCH ' + str(epoch) + '-----')

            if not causal_layer:
                predictions = model(test_embeddings, test_lengths)
            elif causal_layer == 'adversarial':
                predictions, _ = model(test_embeddings, test_lengths)
            elif causal_layer == 'residual':
                predictions, conf_predictions = model(test_embeddings, test_lengths, test_confounders)
            if task == 'classification':
                preds = (predictions.view(-1) >= 0.5).to(device='cpu', dtype=torch.int)
                targets = (test_labels >= 0.5).to(device='cpu', dtype=torch.int)
                accuracy = (preds == targets).sum() * (1 / test_N)
                print('Accuracy on test set: ', accuracy)
                print('Confusion on test set: ', confusion_matrix(targets.numpy(), preds.numpy()))
                print('Precision on test set: ', precision_score(targets.numpy(), preds.numpy()))
                print('Recall on test set: ', recall_score(targets, preds))
                print('F1 on test set: ', f1_score(targets, preds))
                print('Report:\n', classification_report(targets, preds, digits=4))
                print('-----------------')
            else:
                preds = predictions.view(-1).to(device='cpu').detach().numpy()
                targets = test_labels.to(device='cpu', dtype=torch.float).numpy()
                print('RMSE on test set: ', sqrt(mean_squared_error(targets, preds)))
                print('mean pred: ', preds.mean())
                print('mean target: ', targets.mean())

            if causal_layer == 'residual':
                preds = (conf_predictions.view(-1) >= 0.5).to(device='cpu', dtype=torch.int)
                targets = (test_labels >= 0.5).to(device='cpu', dtype=torch.int)
                accuracy = (preds == targets).sum() * (1 / test_N)
                print('Accuracy on test set: ', accuracy)
                print('Confusion on test set: ', confusion_matrix(targets.numpy(), preds.numpy()))
                print('Precision on test set: ', precision_score(targets.numpy(), preds.numpy()))
                print('Recall on test set: ', recall_score(targets, preds))
                print('F1 on test set: ', f1_score(targets, preds))
                print('Report:\n', classification_report(targets, preds, digits=4))
                print('-----------------')

    
    if return_losses:
        if not causal_layer:
            return train_losses, test_losses
        else:
            return train_losses, test_losses, confounder_train_losses, confounder_test_losses
    



def cross_validation_metrics_scores(network, network_params, optimizer_class, loss_fn_class, lr, epochs, batch_size, device,
                                    data, k=5, gru_model=False, shuffle=True):
    """
    :param netwrok: the PyTorch model class
    :param network_params: the arguments of the constructor of the model
    :param optimzer_class: the PyTorch class to used for optimisation
    :param loss_fn_class: the class of the loss function
    :param lr: the learning rate
    :param epochs: the training epochs
    :param batch_size: the batch size of SGD
    :param device: 'cpu' | 'CUDA:0' | ...
    :param data: the data of the model
    :returns the cross validation metrics
    """

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

        valid_input = input[valid_idx, :]
        valid_length = lengths[valid_idx]
        valid_labels = labels[valid_idx]

        train_input = input[train_idx, :]
        train_length = lengths[train_idx]
        train_labels = labels[train_idx]

        train_data = [train_input, train_length, train_labels]
        valid_data = [valid_input, valid_length, valid_labels]
        training_loop_scores(train_data, valid_data, model, device, optimizer, loss_fn, **training_parameters,
                             gru_model=gru_model, verbose=False)
        metrics = Metrics(valid_data, model)
        print(metrics)
        cv_metrics.append(metrics)

        start = end
        end = min(end + samples_in_fold, num_samples)

    print(sum(cv_metrics))
    print(sum(cv_metrics)/k)


def training_loop_scores(data, test_data, model, device, optimizer, loss_fn, epochs=100, batch_size=64, gru_model=False,
                         verbose=True, return_losses=False):
    embeddings, lengths, labels = data
    test_embeddings, test_lengths, test_labels = test_data
    N, input_size = embeddings.shape
    test_N, _ = test_embeddings.shape
    if return_losses:
        train_losses = []
        test_losses = []
    for epoch in range(1, epochs + 1):
        permutation = torch.randperm(N)
        model.train()
        if return_losses:
            train_losses_in_epoch = []
        for i in range(0, N, batch_size):
            optimizer.zero_grad()

            indices = permutation[i:i + batch_size]
            batch_y = labels[indices]
            batch_x = embeddings[indices, :]
            # for data augmentation
            # _, seq_len, _ = batch_x.shape
            # seq_permutation = torch.randperm(seq_len)
            # batch_x = batch_x[:, seq_permutation, :]
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            batch_lengths = lengths[indices]
            batch_lengths = batch_lengths.to(device)

            preds = model(batch_x, batch_lengths).squeeze(1)
            loss = loss_fn(preds, batch_y)
            train_loss = loss.item()
            if return_losses:
                train_losses_in_epoch.append(train_loss)
            loss.backward()
            if gru_model:
                model.hx = model.hx.detach()
            optimizer.step()

        if return_losses:
            model.eval()
            test_preds = model(test_embeddings, test_lengths).squeeze(1)
            test_loss = loss_fn(test_preds, test_labels)
            test_loss = test_loss.item()
            train_loss = np.array(train_losses_in_epoch).mean()
            train_losses.append(test_loss)
            test_losses.append(test_loss)

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
            print('Report:\n', classification_report(targets, preds, digits=4))
            print('-----------------')
            # predictions = model(test_data).squeeze(1)
            # print('RMSE on test set: ', rmse(predictions, test_labels))
    
    if return_losses:
        return train_losses, test_losses
    
