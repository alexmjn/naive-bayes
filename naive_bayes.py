"""Implements Naive Bayes"""
import numpy as np

class NaiveBayes:
    def __init__(self, distribution="Gaussian"):
        self.distribution=distribution
        self.n = None
        self.sd = None
        self.classes = None
        self.class_dict = None
        self.means = None


    def fit(self, distribution):
        pass
