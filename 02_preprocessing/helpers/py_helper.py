# -*- coding: utf-8 -*-
"""Generic Helpers for Python"""
from datetime import datetime
import numpy as np
from collections import OrderedDict

def factors(n):
    return list(set(reduce(list.__add__,
                           ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0))))


def merge_dicts(*dict_args):
    """
    Given any number of dicts, shallow copy and merge into a new dict,
    precedence goes to key value pairs in latter dicts.
    """
    result = {}
    for dictionary in dict_args:
        result.update(dictionary)
    return result

def gcd(a, b): return gcd(b, a % b) if b else a


def convert_date_str_to_info(date_str):
    pit = datetime.strptime(date_str, '%Y-%m-%d')
    return OrderedDict([
        ("month", int(pit.month)),
        ("monthday", int(pit.day)),
        ("weekday", int(pit.strftime('%w'))),
        ("year", int(pit.year)),
        ("yearday", int(pit.strftime('%j'))),
        ("yearweek", int(pit.strftime('%W'))),
    ])


def cartesian_coord(*arrays):
    """feed it with *arrays"""
    grid = np.meshgrid(*arrays)
    coord_list = [entry.ravel() for entry in grid]
    points = np.vstack(coord_list).T
    return points
