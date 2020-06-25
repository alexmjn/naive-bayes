"""Implements Naive Bayes"""
import numpy as np
from gaussian_pandas import gaussian_fit, gaussian_predict

class NaiveBayes:
    def __init__(self, distribution="Gaussian"):
        self.distribution=distribution
        self.n = None
        self.classes = None
        self.class_freqs = None
        self.summary_dfs = None

    def __str__(self):
        print(self.distribution, self.n, self.classes, self.class_freqs,
        self.summary_dfs)

    def fit(self, X, y):
        if self.distribution == "Gaussian":
            gaussian_fit(self, X, y)

    def predict(self, X, y):
        pass
