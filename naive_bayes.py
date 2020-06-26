"""Implements Naive Bayes

This class uses numpy and pandas to implement Naive Bayes with a continuous
Gaussian distribution of variables. Proper usage is to instantiate a model
object with desired hyperparameters, fit the model to a data frame of training
data, then generate predictions."""
import numpy as np
import pandas as pd
from helper import gaussian_numerator

class NaiveBayes:
    def __init__(self, distribution="Gaussian"):
        self.distribution=distribution
        self.classes = []
        self.class_freqs = []
        self.means_sds = []

    def fit(self, X, y):
        """Fits a NaiveBayes() object on supervised training data.

        Takes in a 2-d data frame of feature values and a 1-d data frame or
        Pandas series with the class labels of those features. Calculates
        the class-specific mean and standard deviation for all features
        to be used in Naive Bayes predictions.
        """
        assert isinstance(X, pd.DataFrame), "Please pass a valid data frame."
        assert X.shape[0] == y.shape[0], "Please ensure the dimensions of the inputs match."

        # Get the shape and unique classes of the data
        n = X.shape[0]
        self.classes = y.unique()
        X["target"] = y

        # Separate the data by class, and get the feature distributions.
        for class_type in self.classes:
            df_subclass = X[X["target"] == class_type]
            # Overall class frequency to use as Bayesian prior
            class_freq = df_subclass.shape[0] / n
            self.class_freqs.append(class_freq)

            df_subclass = df_subclass.drop(["target"], axis=1)
            subclass_stds = df_subclass.std()
            subclass_means = df_subclass.mean()
            means_sds = pd.concat([subclass_means, subclass_stds], axis=1)
            self.means_sds.append(means_sds)

        return

    def predict(self, X, y):
        """Use the fitted model object to calculate predictions for new items.

        Takes a data frame of observed feature values and a data frame of targets
        in order to predict the probabilities of classes.
        """
        predictions = []

        # Loop across each unique observation
        for i in range(y.shape[0]):
            class_preds = []

            # Loop across each target class
            for j in range(len(self.classes)):
                probs = self.class_freqs[j]

                # Loop across each feature, getting class-specific data
                for k in range(X.shape[1]):
                    obs = X.iloc[i][k]
                    mean = self.means_sds[j][0][k]
                    sd = self.means_sds[j][1][k]
                    probs = probs * gaussian_numerator(obs, mean, sd)

                class_preds.append(probs)

            # Highest class probability is appended as prediction.
            max_class_index = class_preds.index(max(class_preds))
            predictions.append(self.classes[max_class_index])

            print(class_preds)
        return pd.Series(predictions)



if __name__ == "__main__":
    """ A light test suite."""
    wiki_df = pd.DataFrame()
    wiki_df["height"] = [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75]
    wiki_df["weight"] = [180, 190, 170, 165, 100, 150, 130, 150]
    wiki_df["foot_size"] = [12, 11, 12, 10, 6, 8, 7, 9]

    wiki_target = pd.Series(["male", "male", "male", "male", "female", "female", "female", "female"])

    naive_bayes = NaiveBayes()
    naive_bayes.fit(wiki_df, wiki_target)

    twiki_df = pd.DataFrame()
    twiki_df["height"] = [6, 5]
    twiki_df["weight"] = [130, 95]
    twiki_df["foot_size"] = [8, 3]
    twiki_target = pd.Series(["male", "female"])

    print(naive_bayes.predict(twiki_df, twiki_target))
