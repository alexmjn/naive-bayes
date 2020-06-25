"""Implements Naive Bayes"""
import numpy as np
import pandas as pd
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


if __name__ == "__main__":
    wiki_df = pd.DataFrame()
    wiki_df["height"] = [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75]
    wiki_df["weight"] = [180, 190, 170, 165, 100, 150, 130, 150]
    wiki_df["foot_size"] = [12, 11, 12, 10, 6, 8, 7, 9]

    wiki_target = pd.Series(["male", "male", "male", "male", "female", "female", "female", "female"])

    naive_bayes = NaiveBayes()
    naive_bayes = gaussian_fit(naive_bayes, wiki_df, wiki_target)

    twiki_df = pd.DataFrame()
    twiki_df["height"] = [6]
    twiki_df["weight"] = [130]
    twiki_df["foot_size"] = [8]
    twiki_target = pd.Series(["male", "male"])

    print(gaussian_predict(naive_bayes, twiki_df, twiki_target))
#    [6.197071843878093e-09, 0.0005377909183630022]
