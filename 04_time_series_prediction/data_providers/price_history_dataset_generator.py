from __future__ import division
import csv
import numpy as np
import pandas as pd

from data_providers.price_history_pack import PriceHistoryPack
from price_history_trait import PriceHistoryTrait


class PriceHistoryDatasetGenerator(PriceHistoryTrait):
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
        super(PriceHistoryDatasetGenerator, self).__init__()
        self.random_state = np.random if random_state is None else random_state

    # CSV_IN = '../price_history_02_with_seq_start.csv'
    # NPZ_PATH = '../price_history_02_dp.npz'
    MIN_TARGET_SEQ_LEN = 1
    MIN_INPUT_SEQ_LEN = 2 * MIN_TARGET_SEQ_LEN

    PRICE_HISTORY_SPECIAL_COLS = 2

    def createAndSaveDataset(self, csv_in, input_seq_len, target_seq_len,
                             num_features=1, norm_instance_scale=False,
                             verbose=False, do_global_norm_scale=False,
                             min_input_seq_len=MIN_INPUT_SEQ_LEN, min_target_seq_len=MIN_TARGET_SEQ_LEN,
                             save_files_dic=None,
                             allowSmallerSequencesThanWindow=True, min_date=None, split_fraction=None,
                             keep_training_fraction=None, normalize_targets=False):
        """if min_date is set the price history of all products before this date is neglected
        min_date should be a string of the format YYYY-MM-DD
        """

        dataframe = pd.read_csv(csv_in, index_col=0, quoting=csv.QUOTE_ALL, encoding='utf-8')

        df = self.global_norm_scale(dataframe) if do_global_norm_scale else dataframe

        train_pack, test_pack = self.generate_dataset(norm_instance_scale=norm_instance_scale,
                                                      df=df, input_seq_len=input_seq_len, target_seq_len=target_seq_len,
                                                      num_features=num_features,
                                                      verbose=verbose, min_input_seq_len=min_input_seq_len,
                                                      min_target_seq_len=min_target_seq_len,
                                                      allowSmallerSequencesThanWindow=allowSmallerSequencesThanWindow,
                                                      min_date=min_date,
                                                      normalize_targets=normalize_targets,
                                                      split_fraction=split_fraction)
        if save_files_dic is not None:
            train_pack.save(save_files_dic['train'], fraction=keep_training_fraction, random_state=self.random_state)
            test_pack.save(save_files_dic['test'])

        return train_pack, test_pack

    def generate_dataset(self, df, input_seq_len, target_seq_len,
                         norm_instance_scale=False, num_features=1, verbose=False,
                         min_input_seq_len=MIN_INPUT_SEQ_LEN, min_target_seq_len=MIN_TARGET_SEQ_LEN,
                         allowSmallerSequencesThanWindow=True,
                         min_date=None, split_fraction=None, normalize_targets=False,
                         ):

        assert input_seq_len >= min_input_seq_len
        assert target_seq_len >= min_target_seq_len
        assert min_target_seq_len < min_input_seq_len, \
            "in general we want to sequence of the input be larger than target sequence"

        if verbose:
            print df.shape

        df_reduced = self.removeTooShortSequences(df, target_seq_len=target_seq_len)
        if verbose:
            print df_reduced.shape

        # sku_ids = []
        # XX = np.empty((0, input_seq_len, num_features))
        # YY = np.empty((0, target_seq_len))
        # sequence_lens = []
        # seq_mask = np.empty((0, input_seq_len))
        train_pack = PriceHistoryPack(input_seq_len=input_seq_len, num_features=num_features,
                                      target_seq_len=target_seq_len)
        test_pack = PriceHistoryPack(input_seq_len=input_seq_len, num_features=num_features,
                                     target_seq_len=target_seq_len)

        for is_training, sku_id, inputs, targets in self.getInputsTargets(
                df=df_reduced, input_len=input_seq_len,
                target_len=target_seq_len,
                allowSmallerSequencesThanWindow=allowSmallerSequencesThanWindow,
                min_date=min_date,
                split_fraction=split_fraction,
                normalize_targets=normalize_targets,
                norm_instance_scale=norm_instance_scale,
        ):
            # sku_ids.append(sku_id)
            # inputs_len = len(inputs)
            # sequence_lens.append(inputs_len)
            #
            # # build current mask with zeros and ones
            # cur_mask = np.zeros(input_seq_len)
            # cur_mask[:inputs_len] = 1  # only the valid firsts should have the value of one
            #
            # xx_padded = np.pad(inputs, ((0, input_seq_len - inputs_len), (0, 0)), mode='constant', constant_values=0.)
            # # here targets do NOT need to be padded because we do not have a sequence to sequence model
            # # yy_padded = np.pad(targets, (0, series_max_len - len(targets)), mode='constant', constant_values=0.)
            #
            # assert len(xx_padded) == input_seq_len
            #
            # XX = np.vstack((XX, xx_padded[np.newaxis]))
            # YY = np.vstack((YY, targets[np.newaxis]))
            #
            # seq_mask = np.vstack((seq_mask, cur_mask[np.newaxis]))
            if is_training:
                train_pack.update(sku_id=sku_id, inputs=inputs, targets=targets, input_seq_len=input_seq_len)
            else:
                test_pack.update(sku_id=sku_id, inputs=inputs, targets=targets, input_seq_len=input_seq_len)

        return train_pack, test_pack

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

    def getInputsTargets(self, df, input_len, target_len, normalize_targets, norm_instance_scale,
                         allowSmallerSequencesThanWindow=True, min_date=None, split_fraction=None):
        columns = df.columns

        window_len = input_len + target_len

        for sku_id, row in df.iterrows():
            seq_len = int(row['sequence_length'])

            start_ind = int(row['seq_start'])
            end_ind = start_ind + seq_len

            if min_date is not None and columns[start_ind] < min_date:
                inds_of_min_date = np.argwhere(columns == min_date).flatten()
                if len(
                        inds_of_min_date) == 0:  # this means that min_date was not found at all in our spectrum of dates, this is wrong input
                    raise Exception("please enter a date from {} to {}".format(columns[0], columns[
                        -1 - self.PRICE_HISTORY_SPECIAL_COLS]))
                else:
                    assert len(inds_of_min_date) == 1, "dates must be unique"
                    seq_start = inds_of_min_date[0]
            else:
                seq_start = start_ind

            if seq_start >= end_ind:
                # also this could mean that you have start_ind set to a date after the end of the sequence, this has to be
                # handled as well
                continue  # just skip this instance completely and go to the next

            processed_row = self.instance_scale_normalization(row) if norm_instance_scale else row

            seq = processed_row.iloc[seq_start:end_ind]
            # seq = row_values[seq_start:end_ind]
            # now here we have a sequence and we need to split it in beginning, middle or end randomly

            if split_fraction is None:
                # just return the generator
                for processed_seq in self.processSeq(seq=seq, is_training=True, sku_id=sku_id, window_len=window_len,
                                                     target_len=target_len,
                                                     allowSmallerSequencesThanWindow=allowSmallerSequencesThanWindow,
                                                     normalize_targets=normalize_targets):
                    yield processed_seq
            else:
                max_len = df.shape[1]
                if max_len * split_fraction < window_len:
                    raise Exception("you have used a very small split fraction {} for the window length {} for the"
                                    "price history dataset where the maximum length is {}".format(split_fraction,
                                                                                                  window_len, max_len))

                for is_training, sub_seq in self.split_seq(seq=seq, split_fraction=split_fraction,
                                                           target_seq_len=target_len, window_len=window_len,
                                                           allowSmallerSequencesThanWindow=allowSmallerSequencesThanWindow):
                    for processed_seq in self.processSeq(seq=sub_seq, is_training=is_training, sku_id=sku_id,
                                                         window_len=window_len,
                                                         target_len=target_len,
                                                         allowSmallerSequencesThanWindow=allowSmallerSequencesThanWindow,
                                                         normalize_targets=normalize_targets):
                        yield processed_seq

    def processSeq(self, seq, is_training, sku_id, window_len, target_len, allowSmallerSequencesThanWindow,
                   normalize_targets):
        for seq_window in self.getSeqWindows(
                sequence=seq, seq_length=len(seq),  # seq_len,
                window_length=window_len,
                allowSmallerSequencesThanWindow=allowSmallerSequencesThanWindow):

            if seq_window is None:
                continue
            else:
                seq_norm = self.removeBiasFromSeq(sequence=seq_window.values)

                inputs_flat = seq_norm[:-target_len]

                inputs = inputs_flat[np.newaxis].T

                targets = seq_norm[-target_len:]

                if normalize_targets:
                    last_input = inputs_flat[-1]
                    final_targets = self.removeBiasFromSeq(targets, subtract=last_input)
                else:
                    final_targets = targets

                yield is_training, sku_id, inputs, final_targets

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

    @staticmethod
    def getSeqWindows(sequence, seq_length, window_length, allowSmallerSequencesThanWindow=True):
        """could yield None if smaller sequences than the window are not allowed"""

        # print "length seq"
        # print seq_length
        # indices = np.arange(seq_length, dtype=np.int8)
        # windows = as_strided(indices,  #not a safe function to use
        #                      shape=(seq_length - window_length + 1, window_length),
        #                      strides=(1, 1)) if window_length <= seq_length else indices[np.newaxis]

        if window_length <= seq_length:
            indices = np.arange(seq_length - window_length + 1)
            windows = (indices + np.arange(0, window_length)[:, None]).T
            for window in windows:
                cur_slice = slice(window[0], window[-1] + 1)
                # print "good seq"
                # print sequence[cur_slice]
                yield sequence[cur_slice]
        else:
            # print "sequence len"
            # print len(sequence)
            # print
            # os.system('play --no-show-progress --null --channels 1 synth {} sine {}'.format(0.5, 800))
            if allowSmallerSequencesThanWindow:
                yield sequence
            else:
                yield None

    def removeTooShortSequences(self, df, target_seq_len):
        return df[df['sequence_length'] >= self.getMinimumAllowedSeqLen(target_seq_len=target_seq_len)]

    def getMinimumAllowedSeqLen(self, target_seq_len):
        return target_seq_len * 2

    def split_seq(self, seq, split_fraction, target_seq_len, allowSmallerSequencesThanWindow, window_len):
        """returns whether it is training and the corresponding sequence"""

        min_allowed_len = self.getMinimumAllowedSeqLen(
            target_seq_len=target_seq_len) if allowSmallerSequencesThanWindow else window_len

        cur_seq_len = len(seq)

        part_len = int(np.floor(split_fraction * cur_seq_len))

        min_start = min_allowed_len

        max_start = cur_seq_len - part_len - min_allowed_len

        if min_start < max_start:
            # print "splitted"

            rand_start_pos = self.random_state.randint(min_start, max_start)
            # print rand_start_pos
            end_test_part = rand_start_pos + part_len
            testing_part = seq[rand_start_pos:end_test_part]
            # print testing_part
            train_left = seq[:rand_start_pos]
            train_right = seq[end_test_part:]
            return [
                (True, train_left),
                (False, testing_part),
                (True, train_right),
            ]
        else:
            # print "NOT splitted"
            return [(True, seq)]

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
