# coding: utf-8

from sklearn.linear_model import LogisticRegression
from utils import metrics


class LR(object):
    def __init__(self, penalty='l2', C=100, tol=0.01, class_weight=None, max_iter=100):
        self.classifier = LogisticRegression(
            penalty=penalty,
            C=C,
            tol=tol,
            class_weight=class_weight,
            max_iter=max_iter
        )

    def train(self, X, y):
        self.classifier.fit(X, y)

    def predict(self, X):
        y_pred = self.classifier.predict(X)
        return y_pred

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        precision, recall, f1 = metrics(y_true, y_pred)
        print('Precision: {:.3f}, recall: {:.3f}, F1-measure: {:.3f}\n'.format(precision, recall, f1))
        return precision, recall, f1
