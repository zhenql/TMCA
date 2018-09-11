# Copyright (c) 2018 Ranzhen Li

from . import tools
import numpy as np
from scipy import sparse
import random
import pickle
import os


def hstack(items, num, n_step=7):
    """
    turn SparseVector to ndarray.
    :param items: SparseVector
    :param num: the number of cols.
    :param n_step: the number of rows.
    """
    ret_array = []
    for item in items:
        ret = sparse.csr_matrix((item.datas, (item.rows, item.cols)), shape=(n_step, num))
        ret_array.append(ret.toarray())
    ret_array = np.array(ret_array)
    return ret_array


class SparseVector:
    def __init__(self, sl_list, default_num=None, default_class=None):
        """
        turn sl_list to SparseVector.
        :param sl_list: SparseList
        :param default_num: data which step is smaller than default_num should
            know its context other not.
        :param default_class: data which step is bigger than default_num should be
            give a default_class other than real context.
        """
        if default_num is None:
            default_num = 1000
        if default_class is None:
            default_class = 1000
        cols = np.array([])
        rows = np.array([])
        datas = np.array([])
        for count, item in enumerate(sl_list):
            if count < default_num:
                length = len(item.cols)
                datas = np.r_[datas, [1]*length]
                cols = np.r_[cols, item.cols]
                rows = np.r_[rows, [count]*length]
            else:
                datas = np.r_[datas, [1]]
                cols = np.r_[cols, default_class]
                rows = np.r_[rows, [count]]
        self.datas = datas
        self.cols = cols
        self.rows = rows


class SparseList:
    def __init__(self, cols):
        self.cols = cols


def stringtolist(string):
    tmp = string.split(',')
    ret = []
    for e_idx, e in enumerate(tmp):
        if e_idx == 0:
            if e_idx == len(tmp)-1:
                ret.append(int(e[1:-1]))
            else:
                ret.append(int(e[1:]))
        elif e_idx == len(tmp)-1:
            ret.append(int(e[:-1]))
        else:
            ret.append(int(e))
    return ret
    

class SequentialData:
    def __init__(self, file_dir,
                 config="config.json",
                 split=[0.7, 0.1, 0.2],
                 n_step_decoder=7,
                 default=False,
                 negative=False):
        """
        :param file_dir: the directory of files
            - checkins_u.pkl dict
                key: int(user_id)
                value: list
                    [Int(poi_id), Int(useful), Int(funny), Int(cool), Float(stars),
                    Int(user_id), Int(review_user), Int(funny_user), Int(cool_user), Int(fans),
                    Float(ave_stars), Int(city_id), Int(state_id), Int(business_stars), Int(business_review),
                    Bool(is_open), Float(geo-trans), Float(time-trans), Int(hour), Int(weekday),
                    SparseVector(category), SparseVector(attribute)]
            - checkins.pkl dict
                key: int(user_id)
                value: list
                    [Int(poi_id), String(date), Int(useful), Int(funny), Int(cool),
                    Float(stars), Int(user_id), Int(review_user), Int(useful_user), Int(funny_user),
                    Int(cool_user), Int(fans), Int(ave_stars), Int(city_id), Int(state_id),
                    Float(lat), Float(lon), Int(business_stars), Int(business_review), Bool(is_open),
                    String(category, like"1,2,10"), String(attribute, like"1,2,10")]
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
        self.config, self.data_info = tools.print_config("{}/dapoi/{}".format(file_dir, config),
                                                         default=self.default,
                                                         mode="json")
        self.n_checkins = int(self.config['n_checkins'])
        self.n_users = int(self.config['n_users'])
        self.n_pois = int(self.config['n_pois'])
        self.output_num, self.output_dim = self.get_output_num_dim()
        
        self.data_u = self.get_data_u()
        self.n_data, self.n_train, self.n_val, self.n_test = self.split_data_u()
        self.train_x, self.train_y, self.train_prev_y = self.train()
        
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
            data_s = pickle.load(f)
        data_u = {}
        
        for user in data_s:
            print("=={}/{}==".format(user, self.n_users), end='\r')
            data_u.setdefault(user, [])
            pre_time, pre_poi, pre_lat, pre_lon = (-1, -1, -1, -1)
            for check_idx, check in enumerate(data_s[user]):
                if check_idx == 0:
                    pre_time, pre_poi, pre_lat, pre_lon = (check[1].timestamp(),
                                                           check[0], check[15], check[16])
                cur_time, cur_poi, cur_lat, cur_lon = (check[1].timestamp(),
                                                       check[0], check[15], check[16])
                d_time = cur_time - pre_time
                d_geo = tools.geo_distance(pre_lon, pre_lat, cur_lon, cur_lat)
                weekday = 0 if check[1].weekday() < 5 else 1
                category_sp = SparseList(stringtolist(check[-2]))
                attribute_sp = SparseList(stringtolist(check[-1]))
                tmp1 = np.delete(check, [1, 15, 16, 20, 21])
                tmp2 = np.r_[tmp1, [d_geo, d_time, check[1].hour, weekday, category_sp, attribute_sp]]
                pre_time, pre_poi, pre_lat, pre_lon = (cur_time, cur_poi, cur_lat, cur_lon)
                data_u[user].append(tmp2)
            data_u[user] = np.array(data_u[user])
            
        print("save => " + self.file_dir + "/dapoi/checkins.pkl")
        with open(self.file_dir + "/dapoi/checkins.pkl", "wb") as f:
            pickle.dump(data_u, f)
        return data_u

    def split_data_u(self):
        """
        split dataset.
        """
        print("split data", self.split)
        n_data = {}
        n_train = {}
        n_val = {}
        n_test = {}
        for user in self.data_u:
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
        _x = []
        for j, feature in enumerate(self.data_info):
            if not feature["active"]:
                continue
            if feature["format"] == "embed":
                tmp = np.zeros(dtype=np.int32, shape=(n_size, self.n_step_encoder, 1))
            elif feature["format"] == "interp":
                tmp = np.zeros(dtype=np.float32, shape=(n_size, self.n_step_encoder, 1))
            elif feature["format"] == "int":
                tmp = np.zeros(dtype=np.int32, shape=(n_size, self.n_step_encoder, 1))
            elif feature["format"] == "float":
                tmp = np.zeros(dtype=np.float32, shape=(n_size, self.n_step_encoder, 1))
            elif feature["format"] == "list":
                tmp = np.array([0]*n_size, dtype=object)
            else:
                raise TypeError("no type", feature["format"])
                # tmp = np.zeros(dtype=np.int32, shape=(n_size, self.n_step_encoder, feature["num"]))
            _x.append(tmp)
        _y = np.zeros([n_size], dtype=np.int32)
        _prev_y = np.zeros([n_size, self.n_step_decoder, 1], dtype=np.int32)
        return _x, _y, _prev_y
        
    def next_batch(self, batch_size):
        """
        generate next batch (sample) from train dataset.
        :param batch_size: Int
        """
        index = random.sample(range(0, sum(self.n_train.values())), batch_size)
        index = np.array(index)
        batch_x, batch_y, batch_prev_y = self.init_variable(batch_size)
        for idx_idx, idx in enumerate(index):
            x_index = 0
            for j, feature in enumerate(self.data_info):
                if not feature["active"]:
                    continue
                if feature["format"] == "list":
                    # print("list", self.train_x[x_index][idx])
                    batch_x[x_index][idx_idx] = self.train_x[x_index][idx]
                else:
                    batch_x[x_index][idx_idx, :, :] = self.train_x[x_index][idx, :, :]
                x_index += 1
            batch_y[idx_idx] = self.train_y[idx]
            batch_prev_y[idx_idx] = self.train_prev_y[idx, :, :]
        if self.negative:
            negative_y = np.arange(0, self.output_num)
            np.random.shuffle(negative_y)
            negative_y = negative_y[:batch_size]
            return batch_x, batch_y, batch_prev_y, negative_y
        return batch_x, batch_y, batch_prev_y
    
    def read_dataset(self, mode="train"):
        """
        generate train/val/test dataset.
        :param mode: optional {"train", "val", "test"}
        """
        filename = "{}/dapoi/{}_{}.pkl".format(self.file_dir, mode, self.n_step_decoder)
        if os.path.exists(filename):
            with open(filename, "rb") as f:
                ret = pickle.load(f)
            return ret["x"], ret["y"], ret["prev_y"]
        
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
                x_index = 0
                for feature_idx, feature in enumerate(self.data_info):
                    if not feature["active"]:
                        continue
                    if feature["format"] == "list":
                        default_num, default_class = [None, None]
                        if feature["default"]:
                            default_num = self.n_step_encoder - 1
                            default_class = feature["num"]
                        #  np.array([sparse.csr_matrix((self.n_step_encoder, feature["num"]))]*n_size)
                        x[x_index][cnt] = SparseVector(self.data_u[user][st: xt, feature_idx],
                                                       default_num=default_num,
                                                       default_class=default_class)
                    else:
                        x[x_index][cnt, :self.n_step_encoder, :] = \
                            self.data_u[user][st:xt, feature_idx].reshape(-1, 1)
                        if feature["default"]:
                            if feature["format"] == "embed":
                                x[x_index][cnt, -1, 0] = feature["num"] + 1
                            elif feature["format"] == "float":
                                x[x_index][cnt, -1, 0] = 0
                    x_index += 1
                prev_y[cnt, :, :] = self.data_u[user][st:yt, 0].reshape(-1, 1)
                y[cnt] = self.data_u[user][yt, 0]
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

