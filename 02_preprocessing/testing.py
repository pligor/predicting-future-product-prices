from __future__ import division
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('testing.csv', delimiter=';')

print df.shape
print df.columns
raw_scores = df['raw score'].values[np.newaxis].T

ss = StandardScaler()
print  ss.fit(raw_scores)
print ss.mean_
print ss.std_
scores = ss.transform(raw_scores)
print "go"
print scores.T
print

scores_reshaped = raw_scores.reshape((-1, 4))
print scores_reshaped

means = np.mean(scores_reshaped, axis=1)[np.newaxis].T
print means.shape
stds = np.std(scores_reshaped, axis=1)[np.newaxis].T
print (scores_reshaped - means) / stds


def mydec(value):
    def fundec(fun):
        def fun_wrapper(onlyOneArg):
            return fun(onlyOneArg) * 2 + value
        return fun_wrapper
    return fundec

@mydec(value = 13)
def convertFloatToInt(arg):
    return int(arg)

print convertFloatToInt(2.44)
