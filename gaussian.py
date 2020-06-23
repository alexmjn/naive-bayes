import numpy as np

# assert X and y are proper types
# assert X and y are proportionately shaped
# use y to get prior for predictions
# get sample means and standard deviations (formula for sample,
# # not population s.d)
# self.means = means
# self.sds = sds
# then iterate over the means and sds in the prediction model


def gaussian_fit(naive_bayes, X, y):
    assert isinstance(naive_bayes, NaiveBayes), "Please pass a valid model"
    assert isinstance(X, numpy.ndarray),
    assert X.shape[0] == y.shape[0], "The dimensions of the input do not match."
    naive_bayes.X = X
    naive_bayes.y = y
