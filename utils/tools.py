# coding=utf-8
# ================================
# THIS FILE IS PART OF TMCA
# tools.py - Some tools.
# - add_recall_ndcg: for each batch,
#
# Copyright (C) 2018 Ranzhen Li. All Rights Reserved
# ================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from texttable import Texttable
from math import sin, cos, acos, radians
import numpy as np
import json


def add_recall_ndcg(y_true, y_pred, topk_list):
    ncdg_list = [0]*len(topk_list)
    recall_list = [0] * len(topk_list)
    if y_pred.shape[1] < max(topk_list):
        raise ValueError("y_pred shape is {}, need {}".
                         format(y_pred.shape, (y_pred.shape[0], max(topk_list))))
    for topi in range(max(topk_list)):
        tmp = np.sum(np.equal(y_true, y_pred[:, topi]))
        ncdg = tmp / np.log2(topi+2)
        for topk_idx, topk in enumerate(topk_list):
            if topi < topk:
                ncdg_list[topk_idx] += ncdg
                recall_list[topk_idx] += tmp
    return np.array(recall_list), np.array(ncdg_list)


def geo_distance(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    c = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(lon2 - lon1)
    r = 6371.0
    if c > 1:
        c = 1.0
    return r * acos(c)


def get_time_id(time_format, mode="STELLAR"):
    if mode == "STELLAR":
        if 0 <= time_format.hour < 3:
            time_hour_session = 0
        elif 6 <= time_format.hour < 11:
            time_hour_session = 1
        elif 15 <= time_format.hour < 24:
            time_hour_session = 2
        else:
            time_hour_session = 3
        if time_format.weekday() < 5:
            time_weekday = 0
        else:
            time_weekday = 1
        return time_format.month * 8 + time_weekday*4 + time_hour_session


def print_evaluation(topk_list, recalls=None, ndcgs=None):
    n_cols = len(topk_list)
    cols_align = ["c"] + ["r"] * n_cols
    cols_name = ["dataset"]
    for k in topk_list:
        cols_name.append("@{}".format(k))
    if len(recalls) < 2 * n_cols:
        recalls + [0] * (2 * n_cols - len(recalls))
    if len(ndcgs) < 2 * n_cols:
        ndcgs + [0] * (2 * n_cols - len(ndcgs))
    table = Texttable(0)
    table.set_precision(6)
    table.set_deco(Texttable.HEADER | Texttable.BORDER)
    table.set_cols_align(cols_align)
    table.add_rows([cols_name,
                    ["val_recall"] + recalls[:n_cols],
                    ["test_recall"] + recalls[n_cols:],
                    ["val_ncdg"] + ndcgs[:n_cols],  # update at 2018/4/15/14:49
                    ["test_ncdg"] + ndcgs[n_cols:]])
    print(table.draw())
    return table.draw()


def print_config(filename, default=None, mode="json"):
    if mode == "json":
        print("reading config file:", filename)
        with open(filename, "r") as f:
            config = json.load(f)
        table = Texttable()
        table.set_precision(6)
        table.set_deco(Texttable.BORDER | Texttable.VLINES | Texttable.HEADER)
        
        # for general infomation
        table.set_cols_dtype(['a', 'a'])
        table.set_cols_align(['c', 'c'])
        table.add_rows([["key", "value"]])
        for name_idx, name in enumerate(config.keys()):
            if name == "data info":
                continue
            table.add_row([name, config[name]])
        print(table.draw())
        
        # for data info
        if "data info" in config.keys():
            data_info = config["data info"]
            table = Texttable()
            table.set_deco(Texttable.BORDER | Texttable.VLINES | Texttable.HEADER)
            table.set_cols_dtype(['t', 't', 'a', "a", "a", "a"])
            table.set_cols_align(['c', 'c', 'c', 'c', 'c', 'c'])
            table.add_rows([["name", "format", "number", "dim", "active", "default"]])
            for item in data_info:
                if default is not None:
                    item["default"] = (default and item["default"])
                table.add_row([item["name"], item["format"], item["num"], item["dim"],
                               item["active"], item["default"]])
            print(table.draw())
            return config, data_info
        else:
            return config
    return -1


def print_time(start_, end_):
    run_ = int(end_ - start_)
    run_s = int(run_ % 60)
    run_ = int(run_ / 60)
    run_m = int(run_ % 60)
    run_h = int(run_ / 60)
    # print("{} model runtime: {}:{}:{}".format(name, run_h, run_m, run_s))
    return "{:2d}:{:2d}:{:2d}".format(run_h, run_m, run_s)


def range_small_dataset(total_num, small_size):
    st = 0
    et = min(small_size, total_num)
    while st < total_num:
        yield st, et
        st = et
        et = min(et + small_size, total_num)


