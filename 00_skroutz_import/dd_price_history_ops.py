# -*- coding: UTF-8 -*-
from __future__ import division

import csv
import json
from time import sleep
import pickle
import numpy as np
import pandas as pd
import time
from datetime import datetime
from datetime import timedelta
import os


def printgr(obj):
    print repr(obj).decode('unicode-escape')


class PriceHistoryOps(object):
    PRICE_HISTORY_FOLDER = "price_history"
    DATE_FORMAT = "%Y-%m-%d"
    MOBILES_SKUS_FILENAME = "mobiles_skus"
    LOW_KEY = 'lowest'
    SEQ_LEN_KEY = 'sequence_length'

    def sortAllPriceHistoryAndSave(self):
        skus = np.load("{}.npy".format(self.MOBILES_SKUS_FILENAME))[()]['skus']
        # print len(skus)

        for sku_id in [sku['id'] for sku in skus]:
            ph = self.sortPriceHistory(sku_id=sku_id)
            # print ph
            np.save("{}/{}.npy".format(self.PRICE_HISTORY_FOLDER, sku_id), ph)

    def sortPriceHistory(self, sku_id):
        history = np.load("{}/{}.npy".format(self.PRICE_HISTORY_FOLDER, sku_id))[()]

        for key in [self.LOW_KEY, 'average']:
            history['history'][key] = self.__sortPriceHistory(history=history, key=key)

        return history

    def __sortPriceHistory(self, history, key):
        prices = history['history'][key]

        if type(prices) == dict:
            prices = prices.values()

        sorted_prices = self.__sortPrices(prices=prices)

        return sorted_prices

    def __sortPrices(self, prices):
        return sorted(prices, key=lambda pp: time.strptime(pp['date'], self.DATE_FORMAT))

    def iteratePrices(self, key=LOW_KEY, filename=MOBILES_SKUS_FILENAME):
        skus = np.load("{}.npy".format(filename))[()]['skus']
        # print len(skus)
        for sku_id in [sku['id'] for sku in skus]:
            yield sku_id, np.load("{}/{}.npy".format(self.PRICE_HISTORY_FOLDER, sku_id))[()]['history'][key]

    def isSkusPriceHistoryConsistent(self, sku_id, key=LOW_KEY):
        """note that the prices must be already sorted, there is another method for that in the same class
        and consistent means that from beginning to end there are no missing dates"""
        prices = np.load("{}/{}.npy".format(self.PRICE_HISTORY_FOLDER, sku_id))[()]['history'][key]
        return self.arePricesConsistent(prices=prices)

    def arePricesConsistent(self, prices):
        dates = np.array([datetime.strptime(price['date'], self.DATE_FORMAT) for price in prices])
        if len(dates) > 0:
            prev_date = dates[0]
            for cur_date in dates[1:]:
                check = cur_date == prev_date + timedelta(days=1)
                if not check:
                    return False
                prev_date = cur_date
        return True

    def isPriceHistoryConsistent(self, filename=MOBILES_SKUS_FILENAME, key=LOW_KEY):
        skus = np.load("{}.npy".format(filename))[()]['skus']
        # print len(skus)
        for sku_id in [sku['id'] for sku in skus]:
            check = self.isSkusPriceHistoryConsistent(sku_id=sku_id, key=key)
            if not check:
                return False
        return True

    def makePriceHistoryConsistent(self, filename=MOBILES_SKUS_FILENAME, key=LOW_KEY):
        skus = np.load("{}.npy".format(filename))[()]['skus']
        # print len(skus)
        for sku_id in [sku['id'] for sku in skus]:
            if self.isSkusPriceHistoryConsistent(sku_id=sku_id, key=key):
                pass
            else:
                print "making price history consistent for {}".format(sku_id)
                ph_filepath = "{}/{}.npy".format(self.PRICE_HISTORY_FOLDER, sku_id)

                history = np.load(ph_filepath)[()]

                prices = self.makePricesConsistent(prices=history['history'][key])
                if not self.arePricesConsistent(prices=prices):
                    prices = self.__removePriceHistoryDuplicates(prices=prices)

                assert self.arePricesConsistent(prices=prices)

                history['history'][key] = list(prices)
                # print history
                # break
                np.save(file=ph_filepath, arr=history)

    def removeSkusPriceHistoryDuplicates(self, sku_id, key=LOW_KEY, save=False):
        ph_filepath = "{}/{}.npy".format(self.PRICE_HISTORY_FOLDER, sku_id)
        history = np.load(ph_filepath)[()]
        prices = self.__removePriceHistoryDuplicates(prices=history['history'][key])
        history['history'][key] = prices
        if save:
            np.save(file=ph_filepath, arr=history)
        return prices

    def __removePriceHistoryDuplicates(self, prices):
        """there are no criteria, this is not the most safe operation in the world"""
        dic = dict()
        for price in prices:
            dic[price['date']] = price
        return self.__sortPrices(dic.values())

    def makePricesConsistent(self, prices):
        """consistent means that from beginning to end there are no missing dates"""
        dates = np.array([datetime.strptime(price['date'], self.DATE_FORMAT) for price in prices])
        # print dates
        # print len(prices)

        if len(dates) > 0:
            prev_date = dates[0]
            for ind, cur_date in zip(range(1, len(dates)), dates[1:]):
                check = cur_date == prev_date + timedelta(days=1)
                # print check
                if not check:
                    # print prices[ind-1]
                    # print prices[ind]
                    # print prev_date
                    # print cur_date
                    days_diff = (cur_date - prev_date).days
                    # print days_diff
                    for day_offset in range(1, days_diff):
                        new_date = prev_date + timedelta(days=day_offset)
                        date_str = new_date.strftime(self.DATE_FORMAT)
                        prev_price = prices[ind - 1]
                        new_price = prev_price.copy()
                        new_price['date'] = date_str
                        prices += [new_price]

                prev_date = cur_date

        return np.array(self.__sortPrices(prices=prices))

    def getMinAndMaxDate(self, filename=MOBILES_SKUS_FILENAME, key=LOW_KEY):
        min_date_str = '2100-01-01'
        max_date_str = '2000-01-01'
        for sku_id, prices in self.iteratePrices(filename=filename, key=key):
            # we are considering that the prices are already sorted
            first_date = prices[0]['date']
            last_date = prices[-1]['date']

            if first_date < min_date_str:
                min_date_str = first_date

            if last_date > max_date_str:
                max_date_str = last_date

        return min_date_str, max_date_str

    def getDatesSequence(self, min_date_str, max_date_str):
        """
        # min_date = "2015-08-03"
        # max_date = "2017-06-14"
        """

        assert min_date_str < max_date_str

        min_datetime = datetime.strptime(min_date_str, PriceHistoryOps.DATE_FORMAT)
        max_datetime = datetime.strptime(max_date_str, PriceHistoryOps.DATE_FORMAT)

        # print datetime.strptime(max_date, PriceHistoryOps.DATE_FORMAT)
        delta_dates = max_datetime - min_datetime
        delta_days = delta_dates.days

        dates = []
        for days in xrange(delta_days):
            cur_date = datetime.strftime(min_datetime + timedelta(days=days), self.DATE_FORMAT)
            dates.append(cur_date)

        dates.append(datetime.strftime(max_datetime,
                                       self.DATE_FORMAT))  # also add the last datetime because it is not included with the for loop above

        return dates

    @staticmethod
    def verifyPricesNonZero(prices):
        return np.all([price['price'] > 0 for price in prices])

    def convertPriceHistoriesToDataFrame(self, filename=MOBILES_SKUS_FILENAME):
        min_date_str, max_date_str = self.getMinAndMaxDate(filename=filename)

        date_cols = self.getDatesSequence(min_date_str=min_date_str, max_date_str=max_date_str)

        init_date_price_dict = dict([(date_col, 0) for date_col in date_cols])
        init_date_price_dict[self.SEQ_LEN_KEY] = 0

        dataframe = pd.DataFrame(columns=date_cols)

        for sku_id, prices in pho.iteratePrices():
            cur_date_price_dict = init_date_price_dict.copy()
            # print prices
            assert PriceHistoryOps.verifyPricesNonZero(prices=prices), "all prices make sense to be above zero"

            for price in prices:
                cur_date_price_dict[price['date']] = price['price']

            assert cur_date_price_dict[prices[-1]['date']] > 0, \
                "last price must be non zero, meaning that zeros are allowed only for the past"

            cur_date_price_dict[self.SEQ_LEN_KEY] = len(prices)

            series = pd.Series(data=cur_date_price_dict, name=sku_id)

            dataframe = dataframe.append(series)

        return dataframe


if __name__ == "__main__":
    pho = PriceHistoryOps()

    #pho.makePriceHistoryConsistent()
    #df = pho.convertPriceHistoriesToDataFrame()
    #df.to_csv("../price_history_00.csv", encoding='utf-8', quoting=csv.QUOTE_ALL)
    df = pd.read_csv("../price_history_00.csv", encoding='utf-8', quoting=csv.QUOTE_ALL, index_col=0)

    print len( df[df['sequence_length'] > 30] )

    print "COMPLETED"
    # os.system('play --no-show-progress --null --channels 1 synth {} sine {}'.format(0.5, 800))
