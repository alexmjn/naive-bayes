import numpy as np
import math
from naive_bayes import NaiveBayes
import pandas as pd
from functools import reduce
from helper import gaussian_numerator


def gaussian_fit(naive_bayes, X, y):
    assert isinstance(naive_bayes, NaiveBayes), "Please pass a valid model"
    assert isinstance(X, pd.DataFrame), "Please pass a valid data frame."
    assert X.shape[0] == y.shape[0], "The dimensions of the input do not match."

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

    # drop supervised column
    #  so we can generate sds and means without
    # calculating these over the class column

        df_subclass = df_subclass.drop(["target"], axis=1)
        subclass_stds = df_subclass.std()
        subclass_means = df_subclass.mean()
        summary_df = pd.concat([subclass_means, subclass_stds], axis=1)
        class_summary_dfs.append(summary_df)

    naive_bayes.n = n
    naive_bayes.classes = classes #titles
    naive_bayes.class_freqs = class_freqs #frequences
    naive_bayes.summary_dfs = class_summary_dfs #data frames with sd, mean
    # for each class as a separate df
    return naive_bayes

def gaussian_predict(naive_bayes, X, y):
    """Use the fitted model object to calculate predictions for new items.

    Takes a data frame of observed feature values and a data frame of targets
    in order to predict the probabilities of classes.
    """
    assert isinstance(naive_bayes, NaiveBayes), "Please pass a valid model"
    assert isinstance(X, pd.DataFrame), "Please pass a valid data frame."
    assert X.shape[0] == y.shape[0], "The dimensions of the input do not match."

    all_predictions = []
    for i in range(y.shape[0]):
        class_preds = []

        for j in range(len(naive_bayes.classes)):
            probs = naive_bayes.class_freqs[j]

            for k in range(X.shape[1]):
                obs = X.iloc[i][k]
                mean = naive_bayes.summary_dfs[j][0][k]
                sd = naive_bayes.summary_dfs[j][1][k]
                probs = probs * gaussian_numerator(obs, mean, sd)

            class_preds.append(probs)

        max_class_index = class_preds.index(max(class_preds))
        all_predictions.append(naive_bayes.classes[max_class_index])
    return pd.Series(all_predictions), class_preds

# to do - just initialize probs as the class frequency and multiply each time
# instead of using a list.

# take in integer class labels?

# remove the need to take in y

# print error metrics, at least accuracy
# use if statement to take in y if desired and output accuracy if desired

if __name__ == "__main__":
    wiki_df = pd.DataFrame()
    wiki_df["height"] = [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75]
    wiki_df["weight"] = [180, 190, 170, 165, 100, 150, 130, 150]
    wiki_df["foot_size"] = [12, 11, 12, 10, 6, 8, 7, 9]

    wiki_target = pd.Series(["male", "male", "male", "male", "female", "female", "female", "female"])

    naive_bayes = NaiveBayes()
    naive_bayes = gaussian_fit(naive_bayes, wiki_df, wiki_target)

    twiki_df = pd.DataFrame()
    twiki_df["height"] = [6, 6.5]
    twiki_df["weight"] = [130, 185]
    twiki_df["foot_size"] = [8, 11]
    twiki_target = pd.Series(["male", "male"])

    print(gaussian_predict(naive_bayes, twiki_df, twiki_target))
#    [6.197071843878093e-09, 0.0005377909183630022]
