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
from aa_importing import SkroutzApi

def printgr(obj):
    print repr(obj).decode('unicode-escape')

if __name__ == "__main__":
    pass
    api = SkroutzApi()

    print "COMPLETED"

# print  < time.strptime("2017-02-17", "%Y-%m-%d")
