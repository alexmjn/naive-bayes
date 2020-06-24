import numpy as np
import math
from naive_bayes import NaiveBayes
import pandas as pd
from functools import reduce
# assert X and y are proper types
# assert X and y are proportionately shaped
# use y to get prior for predictions
# get sample means and standard deviations (formula for sample,
# # not population s.d)
# self.means = means
# self.sds = sds
# then iterate over the means and sds in the prediction model


def gaussian_fit_pd(naive_bayes, X, y):
    # assert isinstance(naive_bayes, NaiveBayes), "Please pass a valid model"
    # assert isinstance(X, np.ndarray), "Please pass a valid numpy array."
    # assert X.shape[0] == y.shape[0], "The dimensions of the input do not match."

    # get n, get classes
    n = X.shape[0]
    classes = y.unique()

    # create empty lists to store class frequencies and sds/means
    class_freqs = []
    class_summary_dfs = []
    #this is altering X outside the scope of the function
    X["target"] = y

    # split our df by class -- we need to generate sds and means for each
    # variable for each class.

    for class_type in classes:
        df_subclass = X[X["target"] == class_type]
        class_freq = df_subclass.shape[0] / n
        class_freqs.append(class_freq)

    # drop supervised column so we can generate sds and means without
    # calculating these over the class column

        df_subclass = df_subclass.drop(["target"], axis=1)
        subclass_stds = df_subclass.std()
        subclass_means = df_subclass.mean()
        summary_df = pd.concat([subclass_means, subclass_stds], axis=1)
        class_summary_dfs.append(summary_df)
        print(class_summary_dfs)

    naive_bayes.n = n
    naive_bayes.classes = classes #titles
    naive_bayes.class_freqs = class_freqs #frequences
    naive_bayes.summary_dfs = class_summary_dfs #data frames with sd, mean
    # for each class as a separate df
    return naive_bayes

def gaussian_numerator(obs, mean, sd):
    """Implements probability density from Gaussian distribution

    Takes an observed value of one feature/variable existing in the training set
    and the sample mean and sample standard deviation of that variable as
    calculated in the training set. Plugs that data into the formula for
    normal/Gaussian distribution and returns a posterior probability density
    for
    observing this value for this feature under the assumption of a particular
    class label.

    An example would be: if we're using Naive Bayes to categorize a person as
    male or female, this would be a helper function that generates one part
    of the chain of independent feature posterior densities. Probability of
    male = gaussian_numerator(observedf1, mean of feature 1 in training set,
    sd of f1 in training set) multiplied across all features.
    """
    first_term = 1 / ((2 * math.pi * sd**2)**(1/2))
    exponent = math.exp((-(obs - mean)**2)/(2 * sd**2))
    return first_term * exponent

def gaussian_predict(naive_bayes, X, y):
    # assert isinstance(naive_bayes, NaiveBayes), "Please pass a valid model"
    # assert isinstance(X, np.ndarray), "Please pass a valid numpy array."
    # assert X.shape[0] == y.shape[0], "The dimensions of the input do not match."
# assert classes, dict, n, sd, means exist
# iterate across each feature
  #  for class_cat in naive_bayes.classes:
# generate posterior probability associated with each class
    #    class_prior = class_freq[i]


    # loop across columns. get
    all_predictions = []
    for k in y.shape[0]:
        class_preds = []
        for i in len(naive_bayes.classes):
            probs = []
            probs.append(naive_bayes.class_freqs[i])
            for j in X.shape[1]:
                obs = X[0][i]
                mean = naive_bayes.summary_dfs[i][0][j]
                sd = naive_bayes.summary_dfs[i][1][j]
                probs.append(gaussian_numerator(obs, mean, sd))
            class_prob = reduce(lambda x, y: x*y, probs)
            class_preds.append(class_prob)

        max_class_index = class_preds.index(max(class_preds))
        all_predictions.append(naive_bayes.classes[max_class_index])
    return pd.Series(all_predictions)
