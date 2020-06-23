"""Implements Naive Bayes"""
import numpy as np

class NaiveBayes:
    def __init__(self, distribution=Gaussian):
        self.distribution=distribution

    def fit(self, distribution):
