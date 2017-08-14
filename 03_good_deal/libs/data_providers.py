# -*- coding: utf-8 -*-
"""Data providers.

This module provides classes for loading datasets and iterating over batches of
data points.
"""

import os
import numpy as np
from libs import DATA_DIR
from math_helper import gcd
import pandas as pd
from data_provider import DataProvider, OneOfKDataProvider, CrossValDataProvider
import csv
from sklearn.model_selection import StratifiedShuffleSplit


class ProductPairsDataProvider(OneOfKDataProvider, CrossValDataProvider):
    """Data provider. Note that the pairs are first WORSE (left features) and then BETTER (right features)"""
    PRICE_COLS = ['price_min', 'price_max']
    TARGET_COL = 'price_min'
    BINARY_TARGET_COL = 'binary_target'
    NUM_CLASSES = 2

    def __init__(self, which_set='train', max_num_batches=-1,
                 shuffle_order=True, rng=None, data_dir=os.path.expanduser(DATA_DIR), indices=None):
        """Create a new MNIST data provider object.

        Args:
            which_set: One of 'train', 'valid' or 'test'. Determines which
                portion of the MNIST data this object should provide.
            batch_size (int): Number of data points to include in each batch.
            max_num_batches (int): Maximum number of batches to iterate over
                in an epoch. If `max_num_batches * batch_size > num_data` then
                only as many batches as the data can be split into will be
                used. If set to -1 all of the data will be used.
            shuffle_order (bool): Whether to randomly permute the order of
                the data before each epoch.
            rng (RandomState): A seeded random number generator.
        """

        assert which_set in ['train', 'test'], (
            'Expected which_set to be either train, valid or test. Got {0}'.format(which_set))

        self.which_set = which_set
        self.num_classes = self.NUM_CLASSES

        data_path = os.path.join(data_dir, 'mobiles_05_comp_{}.npz'.format(which_set))
        assert os.path.isfile(data_path), ('Data file does not exist at expected path: ' + data_path)

        # load data from compressed numpy file
        loaded = np.load(data_path)
        inputs, targets, batch_size = loaded['inputs'], loaded['targets'], loaded['batch_size']
        # print type(inputs), inputs.shape
        # print type(targets), targets.shape
        # print type(batch_size), batch_size
        # print targets

        inputs = inputs.astype(np.float32)
        targets = targets.astype(np.int32)
        batch_size = batch_size[()]

        # pass the loaded data to the parent class __init__
        super(ProductPairsDataProvider, self).__init__(
            inputs=inputs, targets=targets, batch_size=batch_size, max_num_batches=max_num_batches,
            shuffle_order=shuffle_order, rng=rng, indices=indices)

    @staticmethod
    def generateInputsAndTargets(which_set, data_dir=os.path.expanduser(DATA_DIR)):
        sets = ['train', 'test']
        assert which_set in sets, (
            'Expected which_set to be either {}. Got {}'.format(sets, which_set))

        df_comp_train = ProductPairsDataProvider.__load_csv(data_dir=data_dir, filename='mobiles_05_comp_train.csv')
        df_comp_test = ProductPairsDataProvider.__load_csv(data_dir=data_dir, filename='mobiles_05_comp_test.csv')
        df_train = ProductPairsDataProvider.__load_csv(data_dir=data_dir, filename='mobiles_03_train.csv')
        df_test = ProductPairsDataProvider.__load_csv(data_dir=data_dir, filename='mobiles_03_test.csv')
        df_data = df_train if which_set == 'train' else pd.concat((df_train, df_test), axis=0)
        df_comp_data = df_comp_train if which_set == 'train' else df_comp_test

        feature_len = df_train.shape[1] * 2
        # self.feature_len = feature_len
        # assert self.feature_len == 278  # yes it includes the minimum and the maximum price inside
        inputs = np.empty((0, feature_len))
        targets = []

        # minor downsampling because we want to have the same batch size on training and testing
        orig_train_size = len(df_comp_train)
        orig_test_size = len(df_comp_test)
        train_size, test_size, batch_size = ProductPairsDataProvider.subsamplingForSameBatchSize(
            train_size=orig_train_size,
            test_size=orig_test_size)
        # print train_size, test_size, batch_size #24940 5742 58

        orig_size = orig_train_size if which_set == 'train' else orig_test_size
        final_size = train_size if which_set == 'train' else test_size

        keep_inds, drop_inds = StratifiedShuffleSplit(n_splits=2, test_size=int(orig_size - final_size)).split(
            X=df_comp_data.drop(labels=[ProductPairsDataProvider.BINARY_TARGET_COL], axis=1),
            y=df_comp_data[ProductPairsDataProvider.BINARY_TARGET_COL]
        ).next()

        sub_df_comp_data = df_comp_data.iloc[keep_inds]

        for ind, tpl in sub_df_comp_data.iterrows():
            worse_id = tpl[0]
            better_id = tpl[1]
            worse = df_data.loc[worse_id].values[np.newaxis]
            better = df_data.loc[better_id].values[np.newaxis]
            paired_features = np.hstack((worse, better))
            inputs = np.vstack((inputs, paired_features))
            targets.append(tpl[2])
            # print paired_features
            # print inputs.shape
            # print worse.shape
            # print better

        targets = np.array(targets)

        return inputs, targets, batch_size

    @staticmethod
    def __load_csv(data_dir, filename):
        data_path = os.path.join(data_dir, filename)  # os.environ['MLP_DATA_DIR']
        assert os.path.isfile(data_path), ('Data file does not exist at expected path: ' + data_path)
        return pd.read_csv(data_path, index_col=0, encoding='utf-8', quoting=csv.QUOTE_ALL)

    @staticmethod
    def subsamplingForSameBatchSize(train_size, test_size,  # train_size=24952, test_size=5868,
                                    threshold=50, depth=10):
        for ii in range(depth):
            for jj in range(depth):
                curgcd = gcd(np.array([train_size]), np.array([test_size]))[0]
                if curgcd >= threshold:
                    return train_size, test_size, curgcd
                test_size -= 2
            train_size -= 2

