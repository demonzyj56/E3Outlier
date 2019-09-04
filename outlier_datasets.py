"""Dataset utilities."""
import numpy as np
import torch
import torch.utils.data
from utils import (
    load_cifar10, load_cifar100, load_mnist, load_fashion_mnist, load_svhn
)




def _load_data_with_outliers(normal, abnormal, p):
    num_abnormal = int(normal.shape[0]*p/(1-p))
    selected = np.random.choice(abnormal.shape[0], num_abnormal, replace=False)
    data = np.concatenate((normal, abnormal[selected]), axis=0)
    labels = np.zeros((data.shape[0], ), dtype=np.int32)
    labels[:len(normal)] = 1
    return data, labels


def _load_data_one_vs_all(data_load_fn, class_ind, p):
    (x_train, y_train), (x_test, y_test) = data_load_fn()
    X = np.concatenate((x_train, x_test), axis=0)
    Y = np.concatenate((y_train, y_test), axis=0)
    normal = X[Y.flatten() == class_ind]
    abnormal = X[Y.flatten() != class_ind]
    return _load_data_with_outliers(normal, abnormal, p)

class OutlierDataset(torch.utils.data.TensorDataset):

    def __init__(self, normal, abnormal, percentage):
        """Samples abnormal data so that the total size of dataset has
        percentage of abnormal data."""
        data, labels = _load_data_with_outliers(normal, abnormal, percentage)
        super(OutlierDataset, self).__init__(
            torch.from_numpy(data), torch.from_numpy(labels)
        )


def load_cifar10_with_outliers(class_ind, p):
    return _load_data_one_vs_all(load_cifar10, class_ind, p)


def load_cifar100_with_outliers(class_ind, p):
    return _load_data_one_vs_all(load_cifar100, class_ind, p)


def load_mnist_with_outliers(class_ind, p):
    return _load_data_one_vs_all(load_mnist, class_ind, p)


def load_fashion_mnist_with_outliers(class_ind, p):
    return _load_data_one_vs_all(load_fashion_mnist, class_ind, p)

def load_svhn_with_outliers(class_ind, p):
    return _load_data_one_vs_all(load_svhn, class_ind, p)
