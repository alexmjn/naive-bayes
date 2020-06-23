"""Implements Naive Bayes"""
import numpy as np

class NaiveBayes:
    def __init__(self, distribution="Gaussian"):
        self.distribution=distribution
        self.n = None
        self.classes = None
        self.class_freqs = None
        self.summary_dfs = None


    def fit(self, distribution):
        pass

    def __str__(self):
        print(self.distribution, self.n, self.classes, self.class_freqs,
        self.summary_dfs)
