"""Implements Naive Bayes"""
import numpy as np
import pandas as pd
from helper import gaussian_numerator

class NaiveBayes:
    def __init__(self, distribution="Gaussian"):
        self.distribution=distribution
        self.classes = []
        self.class_freqs = []
        self.means_sds = []

    def __str__(self):
        print(self.distribution, self.classes, self.class_freqs,
        self.means_sds)

    def fit(self, X, y):
        if self.distribution == "Gaussian":
               # assert isinstance(naive_bayes, NaiveBayes), "Please pass a valid model"
            assert isinstance(X, pd.DataFrame), "Please pass a valid data frame."
        #   assert X.shape[0] == y.shape[0], "The dimensions of the input do not match."

            # get n, get classes
            n = X.shape[0]
            self.classes = y.unique()
            X["target"] = y

            # split our df by class -- we need to generate sds and means for each
            # variable for each class.

            for class_type in self.classes:
                df_subclass = X[X["target"] == class_type]
                class_freq = df_subclass.shape[0] / n
                self.class_freqs.append(class_freq)

            # drop supervised column
            #  so we can generate sds and means without
            # calculating these over the class column

                df_subclass = df_subclass.drop(["target"], axis=1)
                subclass_stds = df_subclass.std()
                subclass_means = df_subclass.mean()
                means_sds = pd.concat([subclass_means, subclass_stds], axis=1)
                self.means_sds.append(means_sds)

            return

    def predict(self, X, y):
        if self.distribution == "Gaussian":
            assert isinstance(X, pd.DataFrame), "Please pass a valid data frame."
            assert X.shape[0] == y.shape[0], "The dimensions of the input do not match."

            all_predictions = []
            for i in range(y.shape[0]):
                class_preds = []

                for j in range(len(self.classes)):
                    probs = self.class_freqs[j]

                    for k in range(X.shape[1]):
                        obs = X.iloc[i][k]
                        mean = self.means_sds[j][0][k]
                        sd = self.means_sds[j][1][k]
                        probs = probs * gaussian_numerator(obs, mean, sd)

                    class_preds.append(probs)

                max_class_index = class_preds.index(max(class_preds))
                all_predictions.append(self.classes[max_class_index])
            return pd.Series(all_predictions), class_preds



if __name__ == "__main__":
    wiki_df = pd.DataFrame()
    wiki_df["height"] = [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75]
    wiki_df["weight"] = [180, 190, 170, 165, 100, 150, 130, 150]
    wiki_df["foot_size"] = [12, 11, 12, 10, 6, 8, 7, 9]

    wiki_target = pd.Series(["male", "male", "male", "male", "female", "female", "female", "female"])

    naive_bayes = NaiveBayes()
    naive_bayes.fit(wiki_df, wiki_target)

    twiki_df = pd.DataFrame()
    twiki_df["height"] = [6]
    twiki_df["weight"] = [130]
    twiki_df["foot_size"] = [8]
    twiki_target = pd.Series(["male"])

    print(naive_bayes.predict(twiki_df, twiki_target))
