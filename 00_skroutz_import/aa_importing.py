# -*- coding: UTF-8 -*-
from __future__ import division

import unirest
import json
from time import sleep
import pickle
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
import os

def printgr(obj):
    print repr(obj).decode('unicode-escape')


class SkroutzApi(object):
    REQUEST_LIMIT = 1000
    DATE_FORMAT = "%Y-%m-%d"
    LIMIT_FILENAME = 'skroutz_limit.txt'

    class RequestLimitExceededException(Exception):
        def __init___(self, dErrorArguments):
            Exception.__init__(self, "request limit exceeded exception raised with args {0}".format(dErrorArguments))
            self.dErrorArguments = dErrorArguments

    class AccessTokenFailedException(Exception):
        def __init___(self, dErrorArguments):
            Exception.__init__(self,
                               "request limit exceeded exception raised with args {0}".format(dErrorArguments))
            self.dErrorArguments = dErrorArguments

    # http://stackoverflow.com/questions/5648573/python-print-unicode-strings-in-arrays-as-characters-not-code-points
    # print repr(leafCategs).decode("unicode-escape")
    MOBILE_PHONES_CATEG = 40
    MOBILES_SKUS_FILENAME = "mobiles_skus"

    LAPTOPS_CATEG = 25
    LAPTOPS_SKUS_FILENAME = "laptops_skus"

    categToFilename = {
        MOBILE_PHONES_CATEG: MOBILES_SKUS_FILENAME,
        LAPTOPS_CATEG: LAPTOPS_SKUS_FILENAME,
    }

    PRICE_HISTORY_FOLDER = "price_history"

    DOCUMENTATION_LINK = "http://developer.skroutz.gr/api/v3/"
    GET_ALL_CATEGORIES = "http://api.skroutz.gr/categories"
    ACCESS_TOKEN_FILENAME = "access_token"
    # ACCEPT_HEADER = "application/vnd.skroutz+json; version=3"
    ACCEPT_HEADER = "application/vnd.skroutz+json; version=3.1"
    GET_FIRST_SKU_CATEG = "http://api.skroutz.gr/categories/{}/skus?per=1"
    GET_CATEG_SPECS = "http://api.skroutz.gr/categories/{}/specifications"
    GET_SKUS_CATEG = "http://api.skroutz.gr/categories/{}/skus?page={}"
    GET_SKUs_SPECS = "http://api.skroutz.gr/skus/{}/specifications"
    GET_SKUs_PRICE_HISTORY = "http://api.skroutz.gr/skus/{}/price_history"
    GET_SKUs_REVIEWS = "http://api.skroutz.gr/skus/{}/reviews?page={}"

    # keep this link safe
    CREDS_LINK = "https://www.skroutz.gr/oauth2/token?client_id=l4awQRZ6iJRsOv05L4E5w==&client_secret=UDhVXEoNkpSQpi47MTBpZChBKSB0sGgwC6XYopRKehzmjxQTkJohbegh9RzyJ4uNJ3RC2V0//q94aZ4fWWvg==&grant_type=client_credentials&scope=public"

    def importData(self, categ_id):
        saved_skus_obj = self.getSavedSkusOfCateg(categ_id=categ_id)

        fresh_skus_obj = self.getAllSkusOfCateg(categId=categ_id)

        merged_skus_dic, new_ids, updated_ids = self.merge_skus(new_skus=fresh_skus_obj['skus'],
                                                                old_skus=saved_skus_obj['skus'])

        skus = merged_skus_dic.values()
        np.save("{}.npy".format(self.categToFilename[categ_id]), {"skus": skus})

        # for new ids retrieve reviews, specs and price history as you would normally do
        self.__retrieveSpecifications(sku_ids=new_ids)
        self.__retrievePriceHistory(sku_ids=new_ids)
        self.__retrieveReviews(sku_ids=new_ids)

        # we update price history of the updated skus every day manually
        self.__updatePriceHistoriesForToday(sku_ids=updated_ids, skus_dic=merged_skus_dic)
        #self.__updatePriceHistories(sku_ids=updated_ids)

        # for updated ids we would NOT need to get the new specs
        # TODO in order to get the new reviews we would need to filter the new

    def __updatePriceHistories(self, sku_ids):
        for sku_id in sku_ids:
            print "updating price history of {}".format(sku_id)
            price_history = self.mergePriceHistories(sku_id=sku_id)
            np.save("{}/{}.npy".format(self.PRICE_HISTORY_FOLDER, sku_id), price_history)

    def __updatePriceHistoriesForToday(self, sku_ids, skus_dic):
        for sku_id in sku_ids:
            self.update_sku_price_history_for_today(sku_id=sku_id, price_min=skus_dic[sku_id]['price_min'])

    def __retrieveSpecifications(self, sku_ids):
        for sku_id in sku_ids:
            print "retrieving specs for {}".format(sku_id)
            specs = self.getSKUsSpecs(sku_id=sku_id)
            np.save("specifications/{}.npy".format(sku_id), specs)
            # specs = np.load("specifications/{}.npy".format(sku_id))[()]
            # print repr(specs).decode('unicode-escape')

    def __retrievePriceHistory(self, sku_ids):
        for sku_id in sku_ids:
            print "retrieving price history for {}".format(sku_id)
            price_history = self.getSKUsPriceHistory(sku_id=sku_id)
            np.save("{}/{}.npy".format(self.PRICE_HISTORY_FOLDER, sku_id), price_history)
            # price_history = np.load("{}/{}.npy".format(self.PRICE_HISTORY_FOLDER, sku_id))[()]
            # print repr(price_history).decode('unicode-escape')

    def __retrieveReviews(self, sku_ids):
        for sku_id in sku_ids:
            print "retrieving reviews for {}".format(sku_id)
            reviews = self.getSKUsReviews(sku_id=sku_id)
            np.save("reviews/{}.npy".format(sku_id), reviews)
            # reviews = np.load("reviews/{}.npy".format(sku_id))[()]
            # print repr(reviews).decode('unicode-escape')

    def getSavedSkusOfCateg(self, categ_id):
        try:
            return np.load(self.categToFilename[categ_id] + ".npy")[()]
        except IOError:
            return {"skus": []}

    def getSKUsReviews(self, sku_id):
        page = 1
        total_pages = 1
        reviews = []
        while page <= total_pages:
            res = self.tryGet(lambda: unirest.get(self.GET_SKUs_REVIEWS.format(sku_id, page),
                                                  headers=self.getAuthHeaders()))

            if res.code == 200:
                body = res.body
                total_pages = body['meta']['pagination']['total_pages']
                reviews += body['reviews']
                print "reviews page {} of sku_id {}".format(page, sku_id)
            else:
                print "page {} could not be retrieved..".format(page)

            page += 1

        return {"reviews": reviews}

    def getSKUsPriceHistory(self, sku_id):
        res = self.tryGet(lambda: unirest.get(self.GET_SKUs_PRICE_HISTORY.format(sku_id),
                                              headers=self.getAuthHeaders()))
        if res.code == 200:
            body = res.body
            print "price history of sku_id {}".format(sku_id)
            return body
        else:
            print "specs of sku {} could not be retrieved..".format(sku_id)
            return None

    def getSKUsSpecs(self, sku_id):
        res = self.tryGet(lambda: unirest.get(self.GET_SKUs_SPECS.format(sku_id),
                                              headers=self.getAuthHeaders()))
        if res.code == 200:
            body = res.body
            print "specifications of sku_id {}".format(sku_id)
            return body
        else:
            print "specs of sku {} could not be retrieved..".format(sku_id)
            return None

    def getAllSkusOfCateg(self, categId):
        page = 1
        total_pages = 1
        skus = []
        while page <= total_pages:
            res = self.tryGet(lambda: unirest.get(self.GET_SKUS_CATEG.format(categId, page),
                                                  headers=self.getAuthHeaders()))

            if res.code == 200:
                body = res.body
                total_pages = body['meta']['pagination']['total_pages']
                print "getting skus page {}".format(page)
                skus += body['skus']
            else:
                print "page {} could not be retrieved..".format(page)

            page += 1
            # if page >= 2:
            #     break

        return {"skus": skus}

    def getSpecsCountPerCateg(self, categs):
        categDf = pd.DataFrame()

        categKeys = categs.keys()
        for categKey in categKeys:
            sleep(0.02)
            res = self.tryGet(lambda: unirest.get(self.GET_CATEG_SPECS.format(categKey),
                                                  headers=self.getAuthHeaders()))

            if res.code == 200:
                print categs[categKey]
                specs_count = len(res.body['specifications'])
                print specs_count
                categDf = categDf.append(
                    {"id": categKey, "name": categs[categKey], "specs_count": specs_count}, ignore_index=True
                )

        return categDf

    def getItemCountPerCateg(self, categs):
        categDf = pd.DataFrame()

        categKeys = categs.keys()
        for categKey in categKeys:
            # sleep(0.01)
            res = self.tryGet(lambda: unirest.get(self.GET_FIRST_SKU_CATEG.format(categKey),
                                                  headers=self.getAuthHeaders()))

            if res.code == 200:
                print categs[categKey]
                # print res.body
                # sleep(0.01)

                total_count = res.body['meta']['pagination']['total_results']
                print total_count
                categDf = categDf.append(
                    {"id": categKey, "name": categs[categKey], "total_count": total_count}, ignore_index=True
                )
                # sleep(0.01)

        return categDf

    def getAllLeafCategs(self):
        leafCategs = {}
        pages = range(1, 96)
        for page in pages:
            res = self.tryGet(lambda: unirest.get("{}?page={}".format(self.GET_ALL_CATEGORIES, page),
                                                  headers=self.getAuthHeaders()))

            curLeafCategs = self.__getLeafCategsOfPage(categs=res.body['categories'])

            leafCategs = dict(curLeafCategs.items() + leafCategs.items())

        return leafCategs

    def __getLeafCategsOfPage(self, categs):
        leafCategs = {}

        for categ in categs:
            if categ['children_count'] == 0:
                leafCategs[categ['id']] = categ['name']

        return leafCategs

    def tryGet(self, callback, retry=True):
        # if self.__checkLimit()
        if self.__check_limit_from_file():
            self.__increaseDailyLimit()
            response = callback()
            if response.code == 200:
                return response
            elif retry:
                self.refreshAccessToken()
                return self.tryGet(callback=callback, retry=False)
            else:
                raise self.AccessTokenFailedException()
        else:
            raise self.RequestLimitExceededException()

    def __check_limit_from_file(self):
        try:
            struct_time, limit = self.__readLimitFromDisk()
            return self.__checkLimit(struct_time=struct_time, limit=limit)
        except IOError:
            self.__resetLimit()
            return True

    def __checkLimit(self, struct_time, limit):
        today = time.mktime(time.strptime(time.strftime(self.DATE_FORMAT), self.DATE_FORMAT))
        if time.mktime(struct_time) < today:
            self.__resetLimit()
            return True
        elif limit < self.REQUEST_LIMIT:
            return True
        else:
            return False

    def __resetLimit(self):
        self.__writeLimit(timestamp=time.time(), cur_requests=0)

    def __increaseDailyLimit(self):
        struct_time, limit = self.__readLimitFromDisk()
        self.__writeLimit(timestamp=time.time(), cur_requests=limit + 1)

    def __writeLimit(self, timestamp, cur_requests):
        with open(self.LIMIT_FILENAME, mode='w') as fp:
            fp.write(datetime.fromtimestamp(timestamp=timestamp).strftime(self.DATE_FORMAT))
            fp.write("\n")
            fp.write(str(cur_requests))

    def __readLimitFromDisk(self):
        # with open(self.LIMIT_FILENAME) as fp:
        #     for line in fp:
        #         print line.strip()

        with open(self.LIMIT_FILENAME) as fp:
            first_line = fp.readline().strip()
            # print "FIRST LINE"
            # print first_line
            struct_time = time.strptime(first_line, self.DATE_FORMAT)
            limit = int(fp.readline().strip())
        return struct_time, limit

    def getAuthHeaders(self):
        with open(self.ACCESS_TOKEN_FILENAME) as tokenfile:
            token = tokenfile.readline()

        return {
            "Accept": self.ACCEPT_HEADER,
            "Authorization": "Bearer {}".format(token)
        }

    def refreshAccessToken(self):
        self.saveCreds(self.authorize())

    def authorize(self):
        """{
            "access_token": "5JUf4VTrRcQ17W_OE-ZSh-LpxPuIBClWjsinAGPHQcHLFo2OjG6vV3d0vC5hEVuI8ejmhWwEHgF72dTg5mq5vQ==",
            "token_type": "bearer",
            "expires_in": 2678399
        }"""
        # return json.loads(unirest.post(self.CREDS_LINK).body)['access_token']
        return unirest.post(self.CREDS_LINK).body['access_token']  # (gets automatically converted to json)

    def saveCreds(self, access_token):
        with open(self.ACCESS_TOKEN_FILENAME, mode='w') as tokenfile:
            tokenfile.write(access_token)

    @staticmethod
    def listToDictionary(the_list):
        the_dic = dict()
        for item in the_list:
            the_dic[item['id']] = item
        return the_dic

    def merge_skus(self, new_skus, old_skus):
        new_skus_dic = self.listToDictionary(new_skus)
        old_skus_dic = self.listToDictionary(old_skus)

        merged_skus_dic = dict(old_skus_dic.items() + new_skus_dic.items())
        new_ids = set(new_skus_dic.keys()).difference(old_skus_dic.keys())
        updated_ids = set(new_skus_dic.keys()).intersection(old_skus_dic.keys())

        return merged_skus_dic, new_ids, updated_ids

    def verifyPriceMinNotGloballyMinimumPrice(self):
        skus = np.load("mobiles_skus.npy")[()]['skus']
        for sku in skus:
            price_min = sku['price_min']
            sku_id = sku['id']
            price_history = np.array([obj['price'] for obj in
                                      np.load("{}/{}.npy".format(self.PRICE_HISTORY_FOLDER, sku_id))[()]['history'][
                                          'lowest']])
            # print price_min, np.min(price_history)
            if np.min(price_history) < price_min:
                return ("we can verify that the price_min attribute is the price of the current date and "
                        "not the overall minimum price since the beginning of time")

        return "not sure"

    def update_sku_price_history_for_today(self, sku_id, price_min):
        filename = "{}/{}.npy".format(self.PRICE_HISTORY_FOLDER, sku_id)
        price_history = np.load(filename)[()]
        lowests = list(np.array(price_history['history']['lowest']).copy())
        old_lowest_len = len(lowests)
        # print repr(lowests).decode('unicode-escape')
        # print datetime.now()
        expanded_lowests = lowests + [{'date': time.strftime(self.DATE_FORMAT),  # today
                                       'price': price_min, 'shop_name': ''}]
        # print repr(expanded_lowests).decode('unicode-escape')

        price_history['history']['lowest'] = expanded_lowests
        assert len(price_history['history']['lowest']) == old_lowest_len + 1
        np.save(filename, price_history)

    def mergePriceHistories(self, sku_id):
        saved_ph = np.load("{}/{}.npy".format(self.PRICE_HISTORY_FOLDER, sku_id))[()]
        online_ph = self.getSKUsPriceHistory(sku_id=sku_id)

        merged_low, low_new_dates, low_updated_dates = self.mergePrices(saved_prices=saved_ph['history']['lowest'],
                                                                        online_prices=online_ph['history']['lowest'])

        merged_avg, avg_new_dates, avg_updated_dates = self.mergePrices(saved_prices=saved_ph['history']['average'],
                                                                        online_prices=online_ph['history']['average'])

        lowest = sorted(merged_low.values(), key=lambda pp: time.strptime(pp['date'], self.DATE_FORMAT))
        average = sorted(merged_avg.values(), key=lambda pp: time.strptime(pp['date'], self.DATE_FORMAT))

        return {"history": {
            'lowest': lowest,
            'average': average,
        }}

        # ph['history']['lowest']

    @staticmethod
    def mergePrices(saved_prices, online_prices):
        low_dic = dict()
        for obj in saved_prices:
            low_dic[obj['date']] = obj

        low_dic_new = dict()
        for obj in online_prices:
            low_dic_new[obj['date']] = obj

        # low_dic_new = low_dic.copy()
        # low_dic_new['2017-04-13'] = {u'date': u'2017-04-13', u'price': 999, u'shop_name': u'Smartphone-repair'}
        # low_dic_new['2017-05-27'] = {u'date': u'2017-05-27', u'price': 0.1, u'shop_name': u'Smartphone-repair'}

        merged_prices = dict(low_dic.items() + low_dic_new.items())
        new_dates = set(low_dic_new.keys()).difference(low_dic.keys())
        updated_dates = set(low_dic_new.keys()).intersection(low_dic.keys())

        return merged_prices, new_dates, updated_dates

if __name__ == "__main__":
    print "hello skroutz"

    SkroutzApi().importData(categ_id=SkroutzApi.MOBILE_PHONES_CATEG)

    print "COMPLETED"
    os.system('play --no-show-progress --null --channels 1 synth {} sine {}'.format(0.5, 800))
# TODO ALWAYS EXECUTE IT OUTSIDE OF PYCHARM
