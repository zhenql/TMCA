# Copyright (c) 2018 Ranzhen Li
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# We disable pylint because we need python3 compatibility.
from texttable import Texttable
from datetime import datetime
import numpy as np
import pandas as pd
import json
import pickle
import os

"""
E:/dataset/gowalla/fromWeb
"""
# TODO add merge function (category from PACE and data from SNAP)


def init_gowalla(gowalla_path="E:/dataset/gowalla/fromWeb/"):
    """
    init gowalla dataset into checkins_u.pkl:
    format:
    checkins_u[int(user_id)].append([int(user_id), int(poi_id), float(lat), float(lon), int(cate_id), time_format])
    sorting!
    :param gowalla_path:
    :return: new_file_name
    """
    poi_feature_file = "poi_feature.csv"
    poi_feature_col_name = ["poi_id", "lat", "lon", "cate_id"]
    checkins_file = "checkins_clear.csv"
    checkins_col_name = ["user_id", "poi_id", "time_str"]
    # category_name_file = "category.csv"
    # category_col_name = ["cate_id", "name"]
    
    print("reading:", gowalla_path + checkins_file)
    checkins_data = pd.read_csv(gowalla_path + checkins_file, header=None, names=checkins_col_name)
    print("reading:", gowalla_path + poi_feature_file)
    poi_feature_data = pd.read_csv(gowalla_path + poi_feature_file, header=None, names=poi_feature_col_name)
    checkins_data = pd.merge(checkins_data, poi_feature_data)
    checkins_data = np.array(checkins_data)
    checkins_u = {}
    for check_idx, check in enumerate(checkins_data):
        print("=={}/{}==".format(check_idx, checkins_data.shape[0]), end="\r")
        [user_id, poi_id, time_str, lat, lon, cate_id] = list(check)
        time_format = datetime.strptime(time_str, " %Y-%m-%dT%H:%M:%SZ")
        checkins_u.setdefault(int(user_id), [])
        checkins_u[int(user_id)].append([int(user_id), int(poi_id), float(lat), float(lon), int(cate_id), time_format])
    print("Sorting...............")
    for user in checkins_u.keys():
        checkins_u[user] = sorted(checkins_u[user], key=lambda x: x[-1])
    new_file_name = "{}/checkins_u.pkl".format(gowalla_path)
    print("save => {}".format(new_file_name))
    with open(new_file_name, "wb") as f:
        pickle.dump(checkins_u, f)
    return new_file_name


def filter_gowalla(gowalla_path="E:/dataset/gowalla/fromWeb/", sequence_len=20, lower_user_num=100, lower_poi_num=100,
                   makesure=False, start_time=None, end_time=None):
    """
    :param gowalla_path:
    :param sequence_len:
    :param lower_user_num:
    :param lower_poi_num:
    :param makesure:
    :param start_time:
    :param end_time:
    :return:
    """
    if not os.path.exists("{}/checkins_u.pkl".format(gowalla_path)):
        init_gowalla(gowalla_path)
    with open("{}/checkins_u.pkl".format(gowalla_path), "rb") as f:
        checkins_u = pickle.load(f)
    
    user_checkin_count = {}
    user_checkin_set = {}
    poi_checkin_set = {}
    cate_checkin_count = {}
    n_checkins = 0
    time_min = datetime.strptime("2014-10-10T12:00:00Z", "%Y-%m-%dT%H:%M:%SZ")
    time_max = datetime.strptime("2000-10-10T12:00:00Z", "%Y-%m-%dT%H:%M:%SZ")
    for user in checkins_u.keys():
        print("=={}==".format(user), end="\r")
        user_checkin_count[user] = len(checkins_u[user])
        user_checkin_set.setdefault(user, set())
        n_checkins += len(checkins_u[user])
        for check_idx, check in enumerate(checkins_u[user]):
            # count
            user_checkin_set[check[0]].add(check[1])
            poi_checkin_set.setdefault(check[1], set())
            poi_checkin_set[check[1]].add(check[0])
            cate_checkin_count.setdefault(check[-2], 0)
            cate_checkin_count[check[-2]] += 1
            # time range
            time_format = check[-1]
            if time_format < time_min:
                time_min = time_format
            if time_format > time_max:
                time_max = time_format
    print("source file config: ")
    config = {"density": n_checkins / (len(user_checkin_set.keys()) * len(poi_checkin_set.keys())),
              "ave of user": n_checkins / len(user_checkin_set.keys()),
              "ave of poi": n_checkins / len(poi_checkin_set.keys()),
              "n_checkins": n_checkins,
              "n_users": len(user_checkin_count.keys()),
              "n_pois": len(poi_checkin_set.keys()),
              "n_category": len(cate_checkin_count.keys()),
              "from": time_min.strftime("%Y-%m-%d %H:%M:%S"),
              "to": time_max.strftime("%Y-%m-%d %H:%M:%S")}
    for config_key in config:
        print(config_key, ":", config[config_key])
    
    old_checkins_u = {}
    # delete with time
    if start_time is not None:
        print("time range: from {} to {}".format(start_time, end_time))
        for user in checkins_u.keys():
            for check_idx, check in enumerate(checkins_u[user]):
                if start_time <= check[-1] <= end_time:
                    old_checkins_u.setdefault(user, [])
                    old_checkins_u[user].append(check)
    else:
        old_checkins_u = checkins_u.copy()
    delete_times = 1
    while True:
        flag = False
        del_user_list = [False] * (max(list(user_checkin_set.keys())) + 1)
        del_poi_list = [False] * (max(list(poi_checkin_set.keys())) + 1)
        for user_id in user_checkin_count.keys():
            if user_checkin_count[user_id] < sequence_len:
                del_user_list[user_id] = True
                flag = True
            if len(user_checkin_set[user_id]) < lower_user_num:
                if makesure or delete_times == 1:
                    del_user_list[user_id] = True
                    flag = True
        for poi_id in poi_checkin_set.keys():
            if len(poi_checkin_set[poi_id]) < lower_poi_num:
                if makesure or delete_times == 1:
                    del_poi_list[poi_id] = True
                    flag = True
        if not flag:
            break
        new_checkins_u = {}
        user_checkin_count = {}
        user_checkin_set = {}
        poi_checkin_set = {}
        for user in old_checkins_u:
            if not del_user_list[user]:
                new_checkins_u.setdefault(user, [])
                user_checkin_set.setdefault(user, set())
                for check_idx, check in enumerate(old_checkins_u[user]):
                    if not del_poi_list[check[1]]:
                        new_checkins_u[user].append(check)
                        user_checkin_set[check[0]].add(check[1])
                        poi_checkin_set.setdefault(check[1], set())
                        poi_checkin_set[check[1]].add(check[0])
                user_checkin_count[user] = len(new_checkins_u[user])
        print("times: {}: #users={}, #POIs={}".format(delete_times,
                                                      len(user_checkin_set.keys()), len(poi_checkin_set.keys())))
        delete_times += 1
        old_checkins_u = new_checkins_u.copy()
    
    # renumber
    print("renumber......")
    user_enum = {}
    poi_enum = {}
    cate_enum = {}
    n_checkins = 0
    new_checkins_u = {}
    time_min = datetime.strptime("2014-10-10T12:00:00Z", "%Y-%m-%dT%H:%M:%SZ")
    time_max = datetime.strptime("2000-10-10T12:00:00Z", "%Y-%m-%dT%H:%M:%SZ")
    for user in old_checkins_u.keys():
        user_enum[user] = len(user_enum.keys())
        new_checkins_u.setdefault(user_enum[user], [])
        n_checkins += len(old_checkins_u[user])
        for check_idx, check in enumerate(old_checkins_u[user]):
            poi_enum.setdefault(check[1], len(poi_enum.keys()))
            new_checkins_u[user_enum[user]].append(check)
            new_checkins_u[user_enum[user]][check_idx][0] = user_enum[user]
            new_checkins_u[user_enum[user]][check_idx][1] = poi_enum[check[1]]
            cate_enum.setdefault(check[-2], len(cate_enum.keys()))
            new_checkins_u[user_enum[user]][check_idx][-2] = cate_enum[check[-2]]
            time_format = check[-1]
            if time_format < time_min:
                time_min = time_format
            if time_format > time_max:
                time_max = time_format
    
    print("Sorting again...............")
    for user in new_checkins_u.keys():
        new_checkins_u[user] = sorted(new_checkins_u[user], key=lambda x: x[-1])
    # save
    if makesure:
        new_file_dir = "{}/{}_{}_{}_1/".format(gowalla_path, sequence_len, lower_user_num, lower_poi_num)
    else:
        new_file_dir = "{}/{}_{}_{}_0/".format(gowalla_path, sequence_len, lower_user_num, lower_poi_num)
    if start_time is not None:
        new_file_dir = "{}{}_{}/".format(new_file_dir, start_time.strftime("%Y-%m-%d"), end_time.strftime("%Y-%m-%d"))
    if not os.path.exists(new_file_dir):
        os.mkdir(new_file_dir)
    print("save into ", new_file_dir)
    with open(new_file_dir + "checkins_u.pkl", "wb") as f:
        pickle.dump(new_checkins_u, f)
    pd.DataFrame({"user": list(user_enum.keys()), "new_id": list(user_enum.values())}) \
        .to_csv(new_file_dir + "user_enum.csv", index=False)
    pd.DataFrame({"poi": list(poi_enum.keys()), "new_id": list(poi_enum.values())}) \
        .to_csv(new_file_dir + "poi_enum.csv", index=False)
    pd.DataFrame({"category": list(cate_enum.keys()), "new_id": list(cate_enum.values())}) \
        .to_csv(new_file_dir + "cate_enum.csv", index=False)
    config = {"density": n_checkins / (len(user_enum.keys()) * len(poi_enum.keys())),
              "ave of user": n_checkins / len(user_enum.keys()),
              "ave of poi": n_checkins / len(poi_enum.keys()),
              "n_checkins": n_checkins,
              "n_users": len(user_enum.keys()),
              "n_pois": len(poi_enum.keys()),
              "n_category": len(cate_enum.keys()),
              "from": time_min.strftime("%Y-%m-%d %H:%M:%S"),
              "to": time_max.strftime("%Y-%m-%d %H:%M:%S")}
    with open(new_file_dir + "config.json", "w") as f:
        json.dump(config, f)
    for key_tmp in config:
        print(key_tmp, ":", config[key_tmp])
    
    # uh = user_history(new_checkins_u, poi_pos=1)
    # print(uh)
    return new_file_dir

if __name__ == '__main__':
    init_gowalla(gowalla_path="E:/dataset/gowalla/fromWeb/")
    filter_gowalla(gowalla_path="E:/dataset/gowalla/fromWeb/",
                   sequence_len=20,
                   lower_user_num=10,
                   lower_poi_num=10,
                   makesure=False, start_time=None, end_time=None)
