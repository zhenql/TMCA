# Copyright (c) 2018 Ranzhen Li
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# We disable pylint because we need python3 compatibility.
from six.moves import xrange
from texttable import Texttable
from datetime import datetime
from . import tools
import numpy as np
import pandas as pd
import random
import pickle
import os

"""
E:/dataset/gowalla/fromWeb
"""


class SequentialData:
    def __init__(self, file_dir="/Dataset/lrz/gowalla/20_20/",
                 config="config.json",
                 split=[0.7, 0.1, 0.2],
                 n_step_decoder=7,
                 default=True,
                 negative=False,
                 upper_time=7 * 24 * 60 * 60,
                 upper_dist=2000):
        """
        :param file_dir: the directory of files
            - checkins_u.pkl dict
                key: int(user_id)
                value: list
                    [int(user_id), int(poi_id), float(lat), float(lon), int(cate_id), time_format]
            - checkins.pkl dict
                key: int(user_id)
                value: list
                    [int(user_id), int(poi_id), float(d_geo), float(d_time),
                    int(hour), int(weekday), int(cate_id)]
            - train_8.pkl (val/test) dict
                key: "x"
                value: [n_train x n_steps_encoder x n_feature_dim]
                key: "y"
                value: [n_train x 1]
                key: "prev_y"
                value: [n_train x n_steps_decoder x 1]
        :param split: [train, validation, test] train + validation + test = 1
        :param n_step_decoder: 7
        :param default: True  =>  if the default is ture in config, give the last input value.
                        False =>  whether the default is ture of false in config, not use the
                        last input value.
        :param negative: TODO
        :param upper_time: the time upper bound.
        :param upper_dist: the distance upper bound.
        """
        self.file_dir = file_dir
        self.default = default
        self.negative = negative
        self.split = split
        self.n_step_decoder = n_step_decoder
        if not self.default:
            self.n_step_encoder = self.n_step_decoder
        else:
            self.n_step_encoder = self.n_step_decoder + 1
        # read config
        self.upper_time = upper_time  # seconds
        self.upper_dist = upper_dist  # (km)
        # TIPS: make config file
        self.config, self.data_info = tools.print_config("{}/dapoi/{}".format(file_dir, config),
                                                         default=self.default, mode="json")
        
        self.output_num, self.output_dim = self.get_output_num_dim()
        self.n_checkins = int(self.config['n_checkins'])
        self.n_users = int(self.config['n_users'])
        self.n_pois = int(self.config['n_pois'])
        
        self.data_u = self.get_data_u()

        self.n_data, self.n_train, self.n_val, self.n_test = self.split_data_u()
        self.train_x, self.train_y, self.train_prev_y = self.train()
        pass
        
    def get_output_num_dim(self):
        """
        get the output_num and output_dim from config.data_info.
        """
        for feature in self.data_info:
            if feature["name"] == "poi":
                output_num = feature["num"]
                output_dim = feature["dim"]
                return output_num, output_dim
        raise ValueError("can not find output in features")
    
    def get_data_u(self):
        """
        read checkins_u.pkl file and then generate checkins.pkl file.
        """
        if os.path.exists(self.file_dir + "/dapoi/checkins.pkl"):
            with open(self.file_dir + "/dapoi/checkins.pkl", "rb") as f:
                return pickle.load(f)
        
        print("reading: ", self.file_dir + "checkins_u.pkl")
        with open(self.file_dir + "checkins_u.pkl", "rb") as f:
            data_us = pickle.load(f)
        data_u = {}
        
        for user in data_us:
            print("=={}/{}==".format(user, self.n_users), end='\r')
            data_u.setdefault(user, [])
            pre_time, pre_poi, pre_lat, pre_lon = (-1, -1, -1, -1)
            for check_idx, check in enumerate(data_us[user]):
                
                # remember the previous visting state. (context)
                if check_idx == 0:
                    pre_time, pre_poi, pre_lat, pre_lon = (check[-1].timestamp(), check[1],
                                                           check[2], check[3])
                cur_time, cur_poi, cur_lat, cur_lon = (check[-1].timestamp(), check[1],
                                                       check[2], check[3])
                
                # calculate the distance and time interval.
                d_time = cur_time - pre_time
                d_geo = tools.geo_distance(pre_lon, pre_lat, cur_lon, cur_lat)
                
                weekday = 0 if check[-1].weekday() < 5 else 1
                
                tmp1 = [d_geo, d_time, check[-1].hour, weekday]
                tmp2 = np.r_[check[0], check[1], tmp1, check[-2]]
                data_u[user].append(tmp2)
                
                pre_time, pre_poi, pre_lat, pre_lon = (check[-1].timestamp(), check[1], check[2], check[3])
            
            data_u[user] = np.array(data_u[user])
            
        print("save => " + self.file_dir + "/dapoi/checkins.pkl")
        with open(self.file_dir + "/dapoi/checkins.pkl", "wb") as f:
            pickle.dump(data_u, f)
        return data_u
    
    def split_data_u(self):
        """
        split dataset.
        """
        print("split data ", self.split)
        n_data = {}
        n_train = {}
        n_val = {}
        n_test = {}
        for user in self.data_u.keys():
            tmp = len(self.data_u[user]) - self.n_step_decoder
            if tmp >= 10:
                n_data[user] = tmp
                n_train[user] = int(n_data[user]*self.split[0])
                n_val[user] = max(int(n_data[user]*self.split[1]), 1)
                n_test[user] = n_data[user] - n_train[user] - n_val[user]
        print("train:", sum(n_train.values()))
        print("validation:", sum(n_val.values()))
        print("test:", sum(n_test.values()))
        return n_data, n_train, n_val, n_test
    
    def init_variable(self, n_size):
        """
        initialize data format.
        :param n_size: batch_size. Int
        """
        batch_x = []
        for j, feature in enumerate(self.data_info):
            if not feature["active"]:
                continue
            if feature["format"] == "embed":
                tmp = np.zeros(dtype=np.int32, shape=(n_size, self.n_step_encoder, 1))
            else:  # interp
                tmp = np.zeros(dtype=np.float32, shape=(n_size, self.n_step_encoder, 1))
            batch_x.append(tmp)
        label = np.zeros([n_size], dtype=np.int32)
        prev_y = np.zeros([n_size, self.n_step_decoder, 1], dtype=np.int32)
        return batch_x, label, prev_y
    
    def next_batch(self, batch_size):
        """
        generate next batch (sample) from train dataset.
        :param batch_size: Int
        """
        index = random.sample(range(0, sum(self.n_train.values())), batch_size)
        index = np.array(index)
        batch_x, true_y, prev_y = self.init_variable(batch_size)
        for cnt, idx in enumerate(index):
            for feature_idx in range(len(self.train_x)):
                batch_x[feature_idx][cnt, :, :] = self.train_x[feature_idx][idx, :, :]
            true_y[cnt] = self.train_y[idx]
            prev_y[cnt] = self.train_prev_y[idx, :, :]
        if self.negative:
            negative_y = np.arange(0, self.output_num, dtype=np.int32)
            np.random.shuffle(negative_y)
            negative_y = negative_y[:batch_size]
            return batch_x, true_y, prev_y, negative_y
        return batch_x, true_y, prev_y
    
    def generate_next_batch(self, batch_size):
        """
        A generater for generating next batch. (continue)
        :param batch_size: Int
        """
        st = 0
        et = batch_size
        n_train = sum(self.n_train.values())
        while True:
            if et > n_train:
                index = np.r_[np.arange(st, n_train), np.arange(0, et - n_train)]
                et = 0
            else:
                index = np.arange(st, et)
            st = et
            et = et + batch_size
            
            batch_x, true_y, prev_y = self.init_variable(batch_size)
            for cnt, idx in enumerate(index):
                for feature_idx in range(len(self.train_x)):
                    batch_x[feature_idx][cnt, :, :] = self.train_x[feature_idx][idx, :, :]
                true_y[cnt] = self.train_y[idx]
                prev_y[cnt] = self.train_prev_y[idx, :, :]
            if self.negative:
                negative_y = np.arange(0, self.output_num, dtype=np.int32)
                np.random.shuffle(negative_y)
                negative_y = negative_y[:batch_size]
                yield batch_x, true_y, prev_y, negative_y
            else:
                yield batch_x, true_y, prev_y

    def read_dataset(self, mode="train"):
        """
        generate train/val/test dataset.
        :param mode: optional {"train", "val", "test"}
        """
        # reading pickle file
        filename = "{}/dapoi/{}_{}.pkl".format(self.file_dir, mode, self.n_step_decoder)
        
        if os.path.exists(filename):
            print("read <=", filename)
            with open(filename, "rb") as f:
                ret = pickle.load(f)
            return ret["x"], ret["y"], ret["prev_y"]
        print("generate", mode, "datasets")
        # mode => n_datasets
        if mode == "train":
            n_datasets = self.n_train
        elif mode == "val":
            n_datasets = self.n_val
        elif mode == "test":
            n_datasets = self.n_test
        else:
            raise ValueError("not exist mode", mode)
        
        index_size = sum(n_datasets.values())
        x, y, prev_y = self.init_variable(index_size)
        # t = np.zeros([index_size, 4], dtype=np.int32)  # save year, month, date, hour
        cnt = 0
        for user in self.n_data.keys():
            print("=={}/{}==".format(user, self.n_users), end='\r')
            # mode => st
            if mode == "train":
                st_add = 0
            elif mode == "val":
                st_add = self.n_train[user]
            else:
                st_add = self.n_train[user] + self.n_val[user]
                
            for i in range(0, n_datasets[user]):
                st = st_add + i
                xt = st + self.n_step_encoder
                yt = st + self.n_step_decoder
                feature_idx = 0
                for j, feature in enumerate(self.data_info):
                    if not feature["active"]:
                        continue
                    x[feature_idx][cnt, :self.n_step_encoder, :] = self.data_u[user][st:xt, j].reshape(-1, 1)
                    if feature["default"]:
                        x[feature_idx][cnt, -1, :] = feature["num"] + 1
                    feature_idx += 1
                prev_y[cnt, :, :] = self.data_u[user][st: yt, 1].reshape(-1, 1)
                y[cnt] = self.data_u[user][yt, 1]
                cnt += 1
        print("save =>", filename)
        with open(filename, "wb") as f:
            pickle.dump({"x": x, "y": y, "prev_y": prev_y}, f)
        return x, y, prev_y
    
    def train(self):
        return self.read_dataset(mode="train")
    
    def validation(self):
        return self.read_dataset(mode="val")
    
    def test(self):
        return self.read_dataset(mode="test")
