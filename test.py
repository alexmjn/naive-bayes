from gaussian_pandas import gaussian_fit_pd, gaussian_predict, gaussian_numerator
from naive_bayes import NaiveBayes
import numpy as np
import pandas as pd
from functools import reduce


wiki_df = pd.DataFrame()
wiki_df["height"] = [6, 5.92, 5.58, 5.92, 5, 5.5, 5.42, 5.75]
wiki_df["weight"] = [180, 190, 170, 165, 100, 150, 130, 150]
wiki_df["foot_size"] = [12, 11, 12, 10, 6, 8, 7, 9]

wiki_target = pd.Series(["male", "male", "male", "male", "female", "female", "female", "female"])

naive_bayes = NaiveBayes()
naive_bayes = gaussian_fit_pd(naive_bayes, wiki_df, wiki_target)

twiki_df = pd.DataFrame()
twiki_df["height"] = [6]
twiki_df["weight"] = [130]
twiki_df["foot_size"] = [8]
twiki_target = pd.Series(["male"])

X = twiki_df
y = twiki_target
all_predictions = []
for k in range(y.shape[0]):
    class_preds = []
    for i in range(len(naive_bayes.classes)):
        probs = []
        probs.append(naive_bayes.class_freqs[i])
        for j in range(X.shape[1]):
            obs = X.iloc[k][i]
            mean = naive_bayes.summary_dfs[i][0][j]
            sd = naive_bayes.summary_dfs[i][1][j]
            probs.append(gaussian_numerator(obs, mean, sd))
        class_prob = reduce(lambda x, y: x*y, probs)
        class_preds.append(class_prob)

    max_class_index = class_preds.index(max(class_preds))
    all_predictions.append(naive_bayes.classes[max_class_index])
