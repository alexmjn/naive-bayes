import numpy as np
import math
from naive_bayes import NaiveBayes
import pandas as pd
# assert X and y are proper types
# assert X and y are proportionately shaped
# use y to get prior for predictions
# get sample means and standard deviations (formula for sample,
# # not population s.d)
# self.means = means
# self.sds = sds
# then iterate over the means and sds in the prediction model


def gaussian_fit(naive_bayes, X, y):
    # assert isinstance(naive_bayes, NaiveBayes), "Please pass a valid model"
    # assert isinstance(X, np.ndarray), "Please pass a valid numpy array."
    # assert X.shape[0] == y.shape[0], "The dimensions of the input do not match."
    n = X.shape[0]
    classes = y.unique()
    class_freqs = []
    class_summary_dfs = []
    X["target"] = y.values
    # split our df by class
    for class_type in classes:
        df_subclass = X[X["target"] == class_type]
        class_freqs = df_subclass.shape[0] / n
        class_freq.append(class_freqs)

        for col in df_subclass.columns:




    sd = np.std(X, axis=1)
    means = np.mean(X, axis=1)
    classes = list(set(y))
    class_probs = []
    for classification in range(len(classes)):
        freq = list(y).count(classification)/(y.shape[0])
        class_probs.append(freq)

    class_dict = dict(zip(classes, class_probs))
    # wrap classes and class probs into a dictionary?

    naive_bayes.classes = classes
    naive_bayes.class_dict = class_dict
    naive_bayes.n = n
    naive_bayes.sd = sd
    naive_bayes.means = means

def gaussian_predict(naive_bayes, X, y):
    assert isinstance(naive_bayes, NaiveBayes), "Please pass a valid model"
    assert isinstance(X, np.ndarray), "Please pass a valid numpy array."
    assert X.shape[0] == y.shape[0], "The dimensions of the input do not match."
# assert classes, dict, n, sd, means exist
# iterate across each feature
    for i in x.shape[1]:
# generate posterior probability associated with each class
