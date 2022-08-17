# coding: utf-8

from sklearn.metrics import precision_recall_fscore_support


def metrics(y_pred, y_true):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")

    return precision, recall, f1
