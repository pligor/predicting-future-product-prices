from __future__ import division
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from mobattrs_price_history_merger import MobAttrsPriceHistoryMerger

# import pandas as pd
# import sys
# import math
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# import re
# import os
# import csv
# from helpers.outliers import MyOutliers
# from skroutz_mobile import SkroutzMobile
# from sklearn.ensemble import IsolationForest
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.tree import DecisionTreeClassifier, export_graphviz
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
# from skroutz_mobile import SkroutzMobile
# from sklearn.model_selection import StratifiedShuffleSplit
# from helpers.my_train_test_split import MySplitTrainTest
# from sklearn.preprocessing import StandardScaler
# from preprocess_price_history import PreprocessPriceHistory
# from price_history import PriceHistory
# from dfa import dfa
# import scipy.signal as ss
# from scipy.spatial.distance import euclidean
# from fastdtw import fastdtw
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import MinMaxScaler
# import random
# from sklearn.metrics import silhouette_score
# from os.path import isfile
# from preprocess_price_history import PreprocessPriceHistory
# from os.path import isfile
# from george import kernels
# import george
# from sklearn.manifold import TSNE


# In[3]:

random_state = np.random.RandomState(seed=16011984)

# In[4]:

window = 3
maxlag = 4
data_path = '../../../../Dropbox/data'
mobattrs_ph_path = data_path + '/mobattrs_price_history'
mobattrs_ph_norm_path = mobattrs_ph_path + '/mobattrs_ph_norm.npy'

# # Gaussian Process Regressor

# In[6]:

arr_norm = np.load(mobattrs_ph_norm_path)

# In[7]:

XX = arr_norm[:, :MobAttrsPriceHistoryMerger.PRICE_IND]

# In[8]:

yy = arr_norm[:, MobAttrsPriceHistoryMerger.PRICE_IND]

# In[9]:
# print "Gaussian Process Regression"
# gp = GaussianProcessRegressor(copy_X_train=False)
# print gp.fit(XX, yy)

#
# # # Dimensionality reduction
#
# # In[10]:
#
from sklearn.decomposition import PCA

pca_reduced = PCA(n_components=84)

XX_reduced = pca_reduced.fit_transform(XX)

# In[34]:
print "Gaussian Process Regression with dim reduct"
gp = GaussianProcessRegressor(copy_X_train=False)
print gp.fit(XX_reduced, yy)

#
# # ## t-SNE dimensionality reduction
# # let's bring on the big guns
#
# # In[ ]:
#
#
#
#
# # # George Approach
#
# # In[26]:
#
# metric = random_state.randn(XX.shape[1], XX.shape[1])
# metric = metric.dot(metric.T)
# metric.shape
#
#
# # In[29]:
#
# kernel = kernels.ExpSquaredKernel(metric=metric, ndim=XX.shape[1])
#
#
# # In[30]:
#
# gp = george.GP(kernel)
# gp
#
#
# # In[31]:
#
# gp.compute(XX)
#
#
# # In[25]:
#
# xx_pred = np.linspace(0, 15, 500)
# xx_pred.shape
#
#
# # In[26]:
#
# pred, pred_var = gp.predict(yy, xx_pred, return_var=True)
#
#
# # In[ ]:
#
#
#
#
# # In[45]:
#
# plt.figure(figsize=(16,8))
# seq.plot()
# plt.plot(preds, 'o')
# plt.show()
#
#
# # In[139]:
#
# #preds - seq.values
#
#
# # # Correlation
#
# # In[256]:
#
# window = 3
#
#
# # In[257]:
#
# seq[:5]
#
#
# # In[258]:
#
# seq.corr(seq[5:10])
#
#
# # In[259]:
#
# seq1 = ph.extractSequenceByLocation(1)
# seq1.shape
#
#
# # In[260]:
#
# seq.corr(seq1)
#
#
# # In[269]:
#
# seqs_tr = [custom_transformation(cur_seq, window=window)[window:] for cur_seq in seqs]
#
#
# # In[270]:
#
# seq_tr = custom_transformation(seq, window)
# seq1_tr = custom_transformation(seq1, window)
#
#
# # In[271]:
#
# seq_tr.corr(seq1_tr)
#
#
# # In[272]:
#
# seq1.plot()
#
#
# # In[273]:
#
# dickey_fuller_print(seq1_tr[seq1_tr==seq1_tr])
#
#
# # In[274]:
#
# corels = np.empty(shape=[len(seqs_tr)] * 2)
# corels.shape
#
#
# # In[276]:
#
# cur_len = len(seqs_tr)
# cur_len
#
#
# # In[279]:
#
# assert np.all([np.all(cur_seq == cur_seq) for cur_seq in seqs_tr]), "we should not have any nan values"
#
#
# # In[280]:
#
# for ii in range(len(seqs_tr)):
#     for jj in range(len(seqs_tr)):
#         corels[ii, jj] = seqs_tr[ii].corr(seqs_tr[jj])
#
#
# # In[281]:
#
# np.save('time_series_correlations.npy', corels)
#
#
# # In[284]:
#
# #aa = np.arange(12).reshape((3,4))
# plt.figure(figsize=(17,17))
# plt.title(
#     'Correlation of transformed and stationary time series of price history of {} mobile phone products'.format(
#         cur_len
#     ))
# sns.heatmap(corels)
# plt.show()
#
#
# # ## Correlation of original time series even though this is prone to errors
# # https://stats.stackexchange.com/questions/133155/how-to-use-pearson-correlation-correctly-with-time-series
#
# # In[290]:
#
# corels = np.empty(shape=[len(seqs)] * 2)
# corels.shape
#
#
# # In[291]:
#
# cur_len = len(seqs)
# cur_len
#
#
# # In[292]:
#
# assert np.all([np.all(cur_seq == cur_seq) for cur_seq in seqs]), "we should not have any nan values"
#
#
# # In[293]:
#
# for ii in range(cur_len):
#     for jj in range(cur_len):
#         corels[ii, jj] = seqs[ii].corr(seqs[jj])
#
#
# # In[294]:
#
# np.save('time_series_orig_correlations.npy', corels)
#
#
# # In[295]:
#
# plt.figure(figsize=(17,17))
# plt.title(
#     'Correlation of the original time series of price history of {} mobile phone products'.format(
#         cur_len
#     ))
# sns.heatmap(corels)
# plt.show()
#
#
# # In[ ]:
#
#
#
