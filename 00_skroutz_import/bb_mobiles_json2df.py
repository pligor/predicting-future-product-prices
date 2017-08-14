# -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import json
import pandas as pd
import os
import csv

def printgr(obj):
    print repr(obj).decode('unicode-escape')


class SkroutzJsonToDataFrame(object):
    SPECS_DIR = 'specifications'

    SPEC_UNIT = {
        u"Τύπος Κινητού": "",
        u"SIM": "",
        u"Λειτουργικό Σύστημα": "",
        u"Έτος Κυκλοφορίας": "",
        u"Ισχύς Βασικού Επεξεργαστή": "GHz",
        u"Πυρήνες Επεξεργαστή": "",
        u"RAM": "GB",
        u"Μνήμη": "GB",
        u"Card Slot": "",
        u"Μέγεθος": "\"",
        u"Ανάλυση": "pixels",
        u"Τύπος": "",
        u"Χειρισμός": "",
        u"Βασική Κάμερα": "MP",
        u"Διπλή Πίσω Κάμερα": "",
        u"Selfie Κάμερα": "MP",
        u"Flash": "",
        u"Δίκτυο Σύνδεσης": "",
        u"Σύνδεσιμότητα": "",
        u"Δείκτης SAR (Head)": "W/kg",
        u"Χωρητικότητα": "mAh",
        u"Αποσπώμενη": "",
        u"Γρήγορη Φόρτιση": "",
        u"Ασύρματη Φόρτιση": "",
        u"Διάρκεια Αναμονής": "hrs",
        u"Διάρκεια Ομιλίας": "hrs",
        u"Προστασία": "",
        u"Αισθητήρες": "",
        u"Διαστάσεις": "mm",
        u"Βάρος": "gr",
    }

    SPEC_CONTROL = u'Χειρισμός'
    SPEC_CONNECTIVITY = u'Σύνδεσιμότητα'
    SPEC_SENSORS = u'Αισθητήρες'
    SPEC_PROTECTION = u'Προστασία',

    def __getMultiValuedSpecsCategories(self):
        return {
            u'Σύνδεσιμότητα': self.CONNECTIVITY_VALUES,
            u'Αισθητήρες': self.SENSOR_VALUES,
            u'Προστασία': self.PROTECTION_VALUES,
            u'Χειρισμός': self.CONTROL_VALUES,
        }

    def __getMultiValuedSpecsToAttrs(self):
        return {
            u'Σύνδεσιμότητα': self.CONNECTIVITY_DICT,
            u'Αισθητήρες': self.SENSOR_DICT,
            u'Προστασία': self.PROTECTION_DICT,
            u'Χειρισμός': self.CONTROL_DICT,
        }

    SPECS_MULTI_VALUED = {
        u'Σύνδεσιμότητα',
        u'Αισθητήρες',
        u'Προστασία',
        u'Χειρισμός',
    }

    def getAllColumns(self):
        arr = self.SPECS_KEYS_DICT.values() + self.SKU_PLAIN_ATTRS + [self.IMAGE_KEY]
        for dic in self.__getMultiValuedSpecsToAttrs().values():
            arr += dic.values()
        return arr

    def createEmptyDic(self):
        dic = dict()
        for key in self.getAllColumns():
            dic[key] = np.NaN
        return dic

    CONNECTIVITY_DICT = {
        u'3.5mm Jack': "connectivity_minijack",
        u'NFC': "connectivity_nfc",
        u'USB': "connectivity_usb",
        u'Bluetooth': "connectivity_bluetooth",
        u'USB (Type-C)': "connectivity_usb_type_c",
        u'Lightning': "connectivity_lightning",
        u'Wi-Fi': "connectivity_wifi",
    }  #7

    SENSOR_DICT = {
        u'Iris Scanner': "sensor_iris_scanner",
        u'Tango': "sensor_tango",
        u'Accelerometer': "sensor_accelerometer",
        u'Proximity': "sensor_proximity",
        u'Δακτυλικό Αποτύπωμα': "sensor_fingerprint",
        u'Hall': "sensor_hall",
        u'Αλτίμετρο': "sensor_altimeter",
        u'Καρδιακών Παλμών': "sensor_heartbeat_meter",
        u'Πυξίδα': "sensor_compass",
        u'Γυροσκόπιο': "sensor_gyroscope",
        u'Light Sensor': "sensor_light",
        u'Βαρόμετρο': "sensor_barometer",
        u'Οξυμετρο': "sensor_oximeter",
    }  #13

    PROTECTION_DICT = {
        u'Dust Resistant': "protection_dust_resistant",
        u'Rugged': "protection_rugged",
        u'Water Resistant': "protection_water_resistant",
    }

    CONTROL_DICT = {
        u'Οθόνη αφής (Touch screen)': "control_touchscreen",
        u'Φυσικό Πληκτρολόγιο': "control_natural_keyboard",
    }

    CONNECTIVITY_VALUES = {u'3.5mm Jack', u'NFC', u'USB', u'Bluetooth', u'USB (Type-C)',
                           u'Lightning', u'Wi-Fi'}

    SENSOR_VALUES = {u'Iris Scanner', u'Tango', u'Accelerometer', u'Proximity', u'Δακτυλικό Αποτύπωμα',
                     u'Hall', u'Αλτίμετρο', u'Καρδιακών Παλμών', u'Πυξίδα', u'Γυροσκόπιο',
                     u'Light Sensor', u'Βαρόμετρο', u'Οξυμετρο'}

    PROTECTION_VALUES = {u'Dust Resistant', u'Rugged', u'Water Resistant'}

    CONTROL_VALUES = {u'Οθόνη αφής (Touch screen)', u'Φυσικό Πληκτρολόγιο'}

    SKU_PLAIN_ATTRS = ['pn', 'name', 'display_name', 'price_max', 'price_min', 'reviewscore', 'shop_count',
                       'plain_spec_summary', 'manufacturer_id', 'future', 'reviews_count', 'virtual', 'id']  #13

    IMAGE_KEY = "main_image"  #1

    SPECS_KEYS_DICT = {
        u'Τύπος Κινητού': 'mobile_type',
        u"SIM": "sim",
        u"Λειτουργικό Σύστημα": "operating_system",
        u"Έτος Κυκλοφορίας": "release_year",
        u"Ισχύς Βασικού Επεξεργαστή": "cpu_power",
        u"Πυρήνες Επεξεργαστή": "cpu_cores",
        u"RAM": "ram",
        u"Μνήμη": "storage",
        u"Card Slot": "card_slot",
        u"Μέγεθος": "diagonal_size",
        u"Ανάλυση": "screen_resolution",
        u"Τύπος": "screen_type",
        u"Βασική Κάμερα": "cam_megapixels",
        u"Διπλή Πίσω Κάμερα": "double_back_cam",
        u"Selfie Κάμερα": "selfie_cam_megapixels",
        u"Flash": "flash",
        u"Δίκτυο Σύνδεσης": "connection_network",
        u"Δείκτης SAR (Head)": "sar",
        u"Χωρητικότητα": "battery_capacity",
        u"Αποσπώμενη": "removable_battery",
        u"Γρήγορη Φόρτιση": "fast_charge",
        u"Ασύρματη Φόρτιση": "wireless_charging",
        u"Διάρκεια Αναμονής": "stanby_autonomy",
        u"Διάρκεια Ομιλίας": "speaking_autonomy",
        u"Διαστάσεις": "dimensions",
        u"Βάρος": "weight",
    }  #26

    SPECS_WITH_EMPTY_VALUES = {u'Προστασία', u'Flash', u'Διάρκεια Ομιλίας', u'Πυρήνες Επεξεργαστή',
                               u'Selfie Κάμερα', u'Card Slot', u'Ανάλυση', u'Διπλή Πίσω Κάμερα',
                               u'Γρήγορη Φόρτιση', u'Δίκτυο Σύνδεσης', u'Χειρισμός', u'RAM',
                               u'Ασύρματη Φόρτιση', u'Βάρος', u'Δείκτης SAR (Head)', u'Βασική Κάμερα',
                               u'Τύπος', u'Ισχύς Βασικού Επεξεργαστή', u'Διάρκεια Αναμονής', u'Σύνδεσιμότητα',
                               u'Διαστάσεις', u'Έτος Κυκλοφορίας', u'Λειτουργικό Σύστημα',
                               u'Αποσπώμενη', u'Αισθητήρες', u'Μνήμη'}

    def __init__(self):
        super(SkroutzJsonToDataFrame, self).__init__()

        self.skus = np.load("mobiles_skus.npy")[()]['skus']

    def skuToDataFrame(self):
        df = pd.DataFrame()

        #max_series_len = 0
        for sku in self.skus:
            sku_id = sku['id']
            print sku_id
            # printgr(sku)

            cur_dic = self.createEmptyDic()

            for sku_attr in self.SKU_PLAIN_ATTRS:
                cur_dic[sku_attr] = sku[sku_attr]

            cur_dic[self.IMAGE_KEY] = sku['images']['main']

            specs = self.getSpecsBySkuId(sku_id=sku_id)
            for spec in specs:
                spec_name = spec['name']
                # print spec_name
                if spec_name in self.SPECS_MULTI_VALUED:
                    categories = self.__getMultiValuedSpecsCategories()[spec_name]
                    specsToAttrs = self.__getMultiValuedSpecsToAttrs()[spec_name]

                    for category in categories:
                        cur_dic[specsToAttrs[category]] = True if category in spec['values'] else False
                else:
                    spec_values = spec['values']
                    cur_dic[self.SPECS_KEYS_DICT[spec_name]] = None if len(spec_values) == 0 else spec_values.pop()
                    assert 0 <= len(spec_values) <= 1

            series = pd.Series(cur_dic, name=sku_id)
            #print len(series)
            # print "len series {}".format(len(series))
            # max_series_len = max(max_series_len, len(series))
            # if len(series) == 65:
            #     pass
            # else:
            #     print series
            assert len(series) == len(self.getAllColumns())
            df = df.append(series)

        # print "max_series_len"
        # print max_series_len

        return df

    def getAllSpecsWithEmptyValues(self):
        spec_names = []
        for sku in self.skus:
            sku_id = sku['id']
            #print sku_id
            specs = self.getSpecsBySkuId(sku_id=sku_id)

            for spec in specs:
                if len(spec['values']) == 0:
                    spec_names.append(spec['name'])

        return set(spec_names)

    def checkIfUnitsSameEverywhere(self, spec_unit):
        for sku in self.skus:
            sku_id = sku['id']
            #print sku_id
            specs = self.getSpecsBySkuId(sku_id=sku_id)

            for spec in specs:
                is_the_same = spec_unit[spec['name']] == spec['unit']
                if not is_the_same:
                    return False
        return True

    def getSpecsBySkuId(self, sku_id):
        return np.load("{}/{}.npy".format(self.SPECS_DIR, sku_id))[()]['specifications']

    def getAllPossibleValues(self, spec_name):
        spec_values = []
        for sku in self.skus:
            sku_id = sku['id']
            # print sku_id
            specs = self.getSpecsBySkuId(sku_id=sku_id)
            # if sku_id == 10705514:
            # printgr(specs)
            # print json.dumps(specs)
            for spec in specs:
                # print spec['name'] in excluded
                # print len(spec['name'])
                if spec['name'] == spec_name:
                    spec_values += spec['values']
        return set(spec_values)

    def checkIfAllSpecsValuesHaveOneOrZeroLength(self, excluded={u'Σύνδεσιμότητα',
                                                                 u'Αισθητήρες',
                                                                 u'Προστασία',
                                                                 u'Χειρισμός',
                                                                 }):
        for sku in self.skus:
            sku_id = sku['id']
            # print sku_id
            specs = np.load("{}/{}.npy".format(self.SPECS_DIR, sku_id))[()]['specifications']
            # if sku_id == 10705514:
            # printgr(specs)
            # print json.dumps(specs)
            for spec in specs:
                # print spec['name'] in excluded
                # print len(spec['name'])
                if spec['name'] in excluded:
                    # print "NOTHING"
                    pass
                else:
                    # print spec['name']
                    # print len(spec['values'])
                    validValues = 0 <= len(spec['values']) <= 1
                    if not validValues:
                        return False
                        # print
                        # break
        return True


if __name__ == "__main__":
    obj = SkroutzJsonToDataFrame()
    dataframe = obj.skuToDataFrame()
    print len(dataframe.columns)
    print len(dataframe)
    dataframe.to_csv("non_processed_mobiles.csv", encoding='utf-8', quoting=csv.QUOTE_ALL)
    os.system('play --no-show-progress --null --channels 1 synth {} sine {}'.format(0.5, 800))
