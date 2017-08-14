# -*- coding: UTF-8 -*-
from __future__ import division
import numpy as np
import pandas as pd
import csv


class SkroutzMobile(object):
    CSV_FILEPATH = "../mobiles_01_preprocessed.csv"

    TEMP_DROP_COLS = ['main_image', 'screen_type', 'display_name']

    PRICE_COLS = ['price_min', 'price_max']
    TARGET_COL = 'price_min'

    BINARY_COLS = [u'wireless_charging', u'removable_battery', u'card_slot', u'double_back_cam', u'fast_charge',
                   u'flash', u'connectivity_bluetooth', u'connectivity_lightning', u'connectivity_minijack',
                   u'connectivity_nfc', u'connectivity_usb', u'connectivity_usb_type_c', u'connectivity_wifi',
                   u'control_natural_keyboard', u'control_touchscreen', u'protection_dust_resistant',
                   u'protection_rugged', u'protection_water_resistant', u'sensor_accelerometer', u'sensor_altimeter',
                   u'sensor_barometer', u'sensor_compass', u'sensor_fingerprint', u'sensor_gyroscope', u'sensor_hall',
                   u'sensor_heartbeat_meter', u'sensor_iris_scanner', u'sensor_light', u'sensor_oximeter',
                   u'sensor_proximity', u'sensor_tango', u'connection_network_0', u'connection_network_1',
                   u'mobile_type_0', u'mobile_type_1', u'manufacturer_id_0', u'manufacturer_id_1', u'manufacturer_id_2',
                   u'manufacturer_id_3', u'manufacturer_id_4', u'manufacturer_id_5', u'manufacturer_id_6',
                   u'manufacturer_id_7', u'manufacturer_id_8', u'manufacturer_id_9', u'manufacturer_id_10',
                   u'manufacturer_id_11', u'manufacturer_id_12', u'manufacturer_id_13', u'manufacturer_id_14',
                   u'manufacturer_id_15', u'manufacturer_id_16', u'manufacturer_id_17', u'manufacturer_id_18',
                   u'manufacturer_id_19', u'manufacturer_id_20', u'manufacturer_id_21', u'manufacturer_id_22',
                   u'manufacturer_id_23', u'manufacturer_id_24', u'manufacturer_id_25', u'manufacturer_id_26',
                   u'manufacturer_id_27', u'manufacturer_id_28', u'manufacturer_id_29', u'manufacturer_id_30',
                   u'manufacturer_id_31', u'manufacturer_id_32', u'manufacturer_id_33', u'manufacturer_id_34',
                   u'manufacturer_id_35', u'manufacturer_id_36', u'manufacturer_id_37', u'manufacturer_id_38',
                   u'manufacturer_id_39', u'manufacturer_id_40', u'manufacturer_id_41', u'manufacturer_id_42',
                   u'manufacturer_id_43', u'manufacturer_id_44', u'manufacturer_id_45', u'manufacturer_id_46',
                   u'manufacturer_id_47', u'manufacturer_id_48', u'manufacturer_id_49', u'manufacturer_id_50',
                   u'manufacturer_id_51', u'manufacturer_id_52', u'manufacturer_id_53', u'manufacturer_id_54',
                   u'manufacturer_id_55', u'manufacturer_id_56', u'manufacturer_id_57', u'manufacturer_id_58',
                   u'manufacturer_id_59', u'manufacturer_id_60', u'manufacturer_id_61', u'manufacturer_id_62',
                   u'manufacturer_id_63', u'manufacturer_id_64', u'manufacturer_id_65', u'manufacturer_id_66',
                   u'manufacturer_id_67', u'manufacturer_id_68', u'manufacturer_id_69', u'manufacturer_id_70',
                   u'operating_system_0', u'operating_system_1', u'operating_system_2', u'operating_system_3']

    def __init__(self):
        super(SkroutzMobile, self).__init__()

        self.df = pd.read_csv(self.CSV_FILEPATH, index_col=0, encoding='utf-8',
                              quoting=csv.QUOTE_ALL).drop(labels=self.TEMP_DROP_COLS, axis=1)

    def getBinaryCols(self):
        binarycols = []
        for col in self.df.columns:
            arr = np.unique(self.df[col])
            if 0 in arr and 1 in arr and len(arr) == 2:
                binarycols.append(col)

        return binarycols


if __name__ == "__main__":
    sm = SkroutzMobile()
    print sm.getBinaryCols()
    print "DONE"
