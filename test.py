from gaussian_pandas import gaussian_fit, gaussian_predict
from helper import gaussian_numerator
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
naive_bayes = gaussian_fit(naive_bayes, wiki_df, wiki_target)

twiki_df = pd.DataFrame()
twiki_df["height"] = [6]
twiki_df["weight"] = [130]
twiki_df["foot_size"] = [8]
twiki_target = pd.Series(["male"])
