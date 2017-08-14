from os import path
import numpy as np
import pandas as pd
from data_providers.price_history_27_dataset_generator import PriceHistory27DatasetGenerator
from relevant_deals import RelevantDeals
import csv
from itertools import islice
from time import sleep


class PriceHistWithRelevantDeals(object):
    yearday = 1
    month_ind = 2
    weekday_ind = 3
    year_ind = 4
    yearweek_ind = 5
    day_ind = 6

    def __init__(self, npz_path, price_history_csv_path, random_state=None, verbose=True):
        super(PriceHistWithRelevantDeals, self).__init__()
        npz_path = npz_path
        assert path.isfile(npz_path)
        self.verbose = verbose

        self.npz = np.load(npz_path)
        self.ph_df = PriceHistory27DatasetGenerator(random_state=random_state).global_norm_scale(
            pd.read_csv(price_history_csv_path, index_col=0, quoting=csv.QUOTE_ALL, encoding='utf-8')
        )

        if verbose:
            for key, val in self.npz.iteritems():
                print key, val.shape

    def execute(self, relevancy_count):
        augmented_inputs_and_inds = list(self.process_inputs(sku_ids=self.npz['sku_ids'], inputs=self.npz['inputs'],
                                                             relevancy_count=relevancy_count))

        augmented_inputs, indices = map(list, zip(*augmented_inputs_and_inds))

        dic = {}
        for key, val in self.npz.iteritems():
            if self.verbose:
                print "processing key: {}".format(key)
            dic[key] = np.array(augmented_inputs) if key == 'inputs' else val[indices]

        return dic

    def process_inputs(self, sku_ids, inputs, relevancy_count):
        for ii, (sku_id, cur_input) in enumerate(zip(sku_ids, inputs)):
            rd = RelevantDeals()

            if sku_id in self.ph_df.index:
                relevant_sku_ids = rd.getSome(sku_id)

                if relevant_sku_ids is None:
                    continue
                else:
                    start_date = self.__get_date_by_item(cur_input[0])
                    end_date = self.__get_date_by_item(cur_input[-1])

                    # https://stackoverflow.com/questions/5234090/how-to-take-the-first-n-items-from-a-generator-or-list-in-python
                    cur_generator = self.process_relevant_sku_ids(relevant_sku_ids=relevant_sku_ids,
                                                                  start_date=start_date, end_date=end_date)
                    sliced_generator = islice(cur_generator, relevancy_count)

                    relevant_inputs = []
                    for relevant_input in sliced_generator:
                        relevant_inputs.append(relevant_input)

                    if len(relevant_inputs) == 0:
                        continue  # we are dropping this input completely

                    rel_inputs_arr = np.array(relevant_inputs).T  # shape (60, 2)

                    if rel_inputs_arr.shape[1] == relevancy_count:
                        augmented = np.hstack((cur_input, rel_inputs_arr))

                        if self.verbose:
                            print "processed input with sku id {} and start date {}, augmented shape {}".format(sku_id,
                                                                                                                start_date,
                                                                                                                augmented.shape)

                        # sleep(0.01)

                        yield augmented, ii
                    else:
                        continue
            else:
                continue

    def process_relevant_sku_ids(self, relevant_sku_ids, start_date, end_date):
        for relevant_sku_id in relevant_sku_ids:
            if relevant_sku_id in self.ph_df.index:
                seq = PriceHistory27DatasetGenerator.extractSequence(self.ph_df.loc[relevant_sku_id])
                check = seq.index[0] <= start_date and end_date <= seq.index[-1]

                if check:
                    begin_ind = np.argwhere(seq.index == start_date).flatten()[0]
                    ending_ind = np.argwhere(seq.index == end_date).flatten()[0]
                    seq_of_interest = seq[begin_ind:ending_ind + 1]

                    unbiased = PriceHistory27DatasetGenerator.removeBiasFromSeq(seq_of_interest)  # shape: 60,
                    ready_deal = unbiased.values  # [np.newaxis].T  # shape (60,)

                    yield ready_deal
                else:
                    continue  # try the next deal
            else:
                continue

    def __get_date_by_item(self, item):
        item = item.astype(np.int32)
        return "{}-{:02d}-{:02d}".format(item[self.year_ind], item[self.month_ind], item[self.day_ind])

    def print_date_info(self, serial_ind):
        print np.unique(self.npz['inputs'][serial_ind][:, 1])  # <--- this is year day
        print np.unique(self.npz['inputs'][serial_ind][:, 2])  # <--- this is month
        print np.unique(self.npz['inputs'][serial_ind][:, 3])  # <--- this is weekday
        print np.unique(self.npz['inputs'][serial_ind][:, 4])  # <---- this is the year
        print np.unique(self.npz['inputs'][serial_ind][:, 5])  # <--- this is year week
        print np.unique(self.npz['inputs'][serial_ind][:, 6])  # <--- this is month day
