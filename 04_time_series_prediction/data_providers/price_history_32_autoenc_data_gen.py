from __future__ import division
import csv
import numpy as np
import pandas as pd

from data_providers.price_history_32_pack import PriceHistoryPack
from mylibs.py_helper import convert_date_str_to_info
from price_history_trait import PriceHistoryTrait


class PriceHistoryAutoEncoderDatasetGenerator(PriceHistoryTrait):
    """
    for splitting dataset to training and testing set there are different ways to do it:
    - We could use as dataset phones we have never seen before
        but this would make our problem a little bit more difficult for our Neural Network
    - Second way is to be stratified. Meaning that we are going to test choosing from all cell phones

    The goal is to choose the golden edge between something that it will not be too hard for our model (since the problem
    of price prediction is already hard by itself) and also that makes sense and gives us the truth about unseen cases.
    So we need to pick a window in time randomly for every product and have this as our testing dataset.

    Another idea is to have products that are close to each other (define distance) and have one
    in training set and the other in testing set. But this does not make full sense since in general we are going to have
    most of the products available. I guess both cases should be tested because we will eventually have new products and
    we will need to know how our algorithm is going to perform on them if only a single window can be generated from the dataset
    """

    def __init__(self, random_state):
        super(PriceHistoryAutoEncoderDatasetGenerator, self).__init__()
        self.random_state = np.random if random_state is None else random_state

    PRICE_HISTORY_SPECIAL_COLS = 2

    def createAndSaveDataset(self, csv_in,
                             norm_instance_scale=False,
                             verbose=False, do_global_norm_scale=False,
                             save_files_dic=None,
                             split_fraction=None,
                             keep_training_fraction=None):

        dataframe = pd.read_csv(csv_in, index_col=0, quoting=csv.QUOTE_ALL, encoding='utf-8')

        df = self.global_norm_scale(dataframe) if do_global_norm_scale else dataframe

        train_pack, test_pack = self.generate_dataset(norm_instance_scale=norm_instance_scale,
                                                      df=df,
                                                      verbose=verbose,
                                                      split_fraction=split_fraction)

        if save_files_dic is not None and save_files_dic['train'] is not None:
            train_pack.save(save_files_dic['train'], fraction=keep_training_fraction, random_state=self.random_state)

        if save_files_dic is not None and save_files_dic['test'] is not None:
            test_pack.save(save_files_dic['test'])

        return train_pack, test_pack

    def generate_dataset(self, df,
                         norm_instance_scale=False, verbose=False,
                         split_fraction=None
                         ):
        """by convention the rows are the instances and the columns are the dates of the time series
        Each time series extends along a row from first to last column"""

        if verbose:
            print df.shape

        train_pack = PriceHistoryPack()
        test_pack = PriceHistoryPack()

        max_seq_len = df.shape[1] - self.PRICE_HISTORY_SPECIAL_COLS

        for is_training, sku_id, inputs, date_inputs in self.getProcessedSeqs(
                df=df,
                split_fraction=split_fraction,
                norm_instance_scale=norm_instance_scale,
        ):
            if is_training:
                train_pack.update(sku_id=sku_id, inputs=inputs, max_seq_len=max_seq_len, date_inputs=date_inputs)
            else:
                test_pack.update(sku_id=sku_id, inputs=inputs, max_seq_len=max_seq_len, date_inputs=date_inputs)

        return train_pack, test_pack

    def getProcessedSeqs(self, df, norm_instance_scale, split_fraction=None):
        for sku_id, row in df.iterrows():
            seq_len = int(row['sequence_length'])

            start_ind = int(row['seq_start'])
            end_ind = start_ind + seq_len

            seq_start = start_ind

            assert seq_start < end_ind

            processed_row = self.instance_scale_normalization(row) if norm_instance_scale else row

            seq = processed_row.iloc[seq_start:end_ind]
            # seq = row_values[seq_start:end_ind]
            # now here we have a sequence and we need to split it in beginning, middle or end randomly

            if split_fraction is None:
                yield self.processSeq(seq=seq, is_training=True, sku_id=sku_id)
            else:
                raise NotImplementedError  # TODO to implement this later

    def processSeq(self, seq, is_training, sku_id):
        seq_norm = self.removeBiasFromSeq(sequence=seq.values)
        dates = seq.index
        return is_training, sku_id, seq_norm, dates

    @staticmethod
    def removeBiasFromSeq(sequence, subtract=None):
        # import warnings
        # warnings.filterwarnings('error')

        if subtract is None:
            # try:
            return sequence - sequence[0]
            # except RuntimeWarning:
            #     raise Exception("sequence[0] {} \n seq: {}".format(sequence[0], sequence))
        else:
            return sequence - subtract

    def global_norm_scale(self, df):
        seqs = self.extractAllSequences(df=df)
        vals = []
        for seq in seqs:
            vals += list(seq.values)

        global_std = np.std(vals)

        if global_std == 0:
            raise Exception("the global std of all time series is zero. This is clearly something wrong")

        new_dataframe = df.copy()
        val_cols = df.columns[:-self.PRICE_HISTORY_SPECIAL_COLS]
        new_dataframe[val_cols] = df[val_cols] / global_std

        return new_dataframe

    @staticmethod
    def create_subsampled(inpath, target_size, outpath, random_state):
        dic = np.load(inpath)
        original_size = len(dic[dic.keys()[0]])
        random_inds = random_state.choice(original_size, target_size, replace=False)
        subsampled_dic = {}
        for key in dic.keys():
            subsampled_dic[key] = dic[key][random_inds]
        np.savez(outpath, **subsampled_dic)

    @staticmethod
    def train_test_split(fullpath, test_size, train_path, test_path, random_state):
        dic = np.load(fullpath)
        original_size = len(dic[dic.keys()[0]])
        test_inds = random_state.choice(original_size, test_size, replace=False)
        train_inds = list(set(range(original_size)).difference(test_inds))

        train_dic = {}
        test_dic = {}
        for key in dic.keys():
            train_dic[key] = dic[key][train_inds]
            test_dic[key] = dic[key][test_inds]

        np.savez(train_path, **train_dic)
        np.savez(test_path, **test_dic)

    @staticmethod
    def merge_date_info(npz_path):
        dic = np.load(npz_path)
        # keys = {'sku_ids', 'inputs', 'targets', 'sequence_lengths', 'sequence_masks', 'date_inputs', 'date_targets'}
        # assert set(dic.keys()) == keys, "npz should contain all of these keys: {}".format(keys)
        necessary_keys = {'inputs', 'date_inputs'}
        assert len(
            necessary_keys.difference(set(dic.keys()))) == 0, "npz should contain at least these keys: {}".format(
            necessary_keys)

        XX = []
        extra_inputs = []
        for cur_input, cur_date_stream, cur_seq_len in zip(dic['inputs'], dic['date_inputs'], dic['sequence_lengths']):
            date_list = []
            for ii, cur_date in enumerate(cur_date_stream):
                converted = convert_date_str_to_info(cur_date).values() if ii < cur_seq_len else [-1] * 6
                date_list.append(converted)

            date_info = np.array(date_list)

            extra_inputs.append(date_info)

            shaped_in = cur_input[np.newaxis].T
            full_info = np.concatenate((shaped_in, date_info), axis=1)
            XX.append(full_info)

        copydic = {}
        for key, val in dic.iteritems():
            copydic[key] = val

        copydic['inputs'] = np.array(XX)  # overwrite inputs
        copydic['extra_inputs'] = np.array(extra_inputs)
        del copydic['date_inputs']  # erase merged info

        return copydic

    @staticmethod
    def normalize_date_info(npz_path):
        dic = np.load(npz_path)
        # keys = {'sku_ids', 'inputs', 'targets', 'sequence_lengths', 'sequence_masks', 'date_inputs', 'date_targets'}
        # assert set(dic.keys()) == keys, "npz should contain all of these keys: {}".format(keys)
        necessary_keys = {'inputs', 'extra_inputs'}
        assert len(
            necessary_keys.difference(set(dic.keys()))) == 0, "npz should contain at least these keys: {}".format(
            necessary_keys)

        ind_dic = {
            "month": 0,
            "monthday": 1,
            "weekday": 2,
            "year": 3,
            "yearday": 4,
            "yearweek": 5,
        }

        copydic = {}
        for key, val in dic.iteritems():
            copydic[key] = val

        copydic['extra_inputs'] = copydic['extra_inputs'].astype(np.float32)

        for key, ind in ind_dic.iteritems():
            date_vals = copydic['extra_inputs'][:, :, ind].astype(np.float32)
            date_vals_flat = date_vals.flatten()
            date_vals_filtered = date_vals_flat[date_vals_flat >= 0]
            cur_mean = np.mean(date_vals_filtered)
            cur_std = np.std(date_vals_filtered)

            for ii, cur_len in enumerate(copydic['sequence_lengths']):
                date_vals[ii, :cur_len] = (date_vals[ii, :cur_len] - cur_mean) / cur_std

            copydic['extra_inputs'][:, :, ind] = date_vals

        copydic['inputs'][:, :, 1:] = copydic['extra_inputs']

        return copydic

    def instance_scale_normalization(self, item):
        """normalize the scale of a row of the dataframe"""
        seq = self.extractSequence(row=item)
        std = np.std(seq)
        if std == 0:
            return item
        else:
            newitem = item.copy()
            cur_slice = slice(0, -self.PRICE_HISTORY_SPECIAL_COLS)
            newitem.iloc[cur_slice] = item.iloc[cur_slice] / std

            return newitem
