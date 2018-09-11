# coding=utf-8
# ================================
# THIS FILE IS PART OF TMCA
# TMCA.py - The running part of the TMCA model
#
# Copyright (C) 2018 Ranzhen Li. All Rights Reserved
# ================================
import time
import numpy as np
import tensorflow as tf
import pandas as pd
from texttable import Texttable
from utils import GenerateGowallaInput, tools, GenerateYelpInput
from utils import TMCAModel
from utils.GenerateYelpInput import hstack

# ===== hyper parameters ===========
series_step = 7  # 7 9 11 19 29 39 49
hidden_units = 60  # 20 40 60
# 3 6 12 24 48 72 96 120 144 168 192
dtime_thershold = 3  # hours
# 1 3 5 10 15 20 25 30
dgeo_thershold = 10  # kilometers
reg = 0
learning_rate = 0.001
epochs = 30000
display_step = 3000
train_size = 256

# =========== settings =============
max_to_keep = 1
sample = 0
topk_list = [1, 2, 5, 10, 15, 20, 25, 30, 35, 40]
opt_method = "Adam"
method_name = "TMCA"
dir_name = "20_10_10_0"
dataset_name = "gowalla"
rnn = "LSTM"
directory = "/home/liranzhen/dataset/{}/{}/".format(dataset_name, dir_name)
config = "config602.json"
log_path = "./result/{0}/{1}/{2}/{2}_{3}/".format(dataset_name, dir_name, method_name, time.strftime("%Y%m%d_%H%M"))
val_path = log_path + "validation/"
test_path = log_path + "test/"
summary_path = log_path + "summary/"

# ===== other parameters ===========
n_steps_encoder = series_step + 1
n_steps_decoder = series_step
n_hidden_encoder = hidden_units
n_hidden_decoder = hidden_units
if method_name == "TMCA_noattn":
    frag_mode = [False, False]
elif method_name == "TMCA_context":
    frag_mode = [True, False]
elif method_name == "TMCA_temporal":
    frag_mode = [False, True]
else:
    frag_mode = [True, True]
# ========================================== read dataset ===========================================
print("read dataset")
if dataset_name == "gowalla":
    DataInput = GenerateGowallaInput.SequentialData
else:
    DataInput = GenerateYelpInput.SequentialData

DATA = DataInput(file_dir=directory,
                 config=config,
                 split=[0.7, 0.1, 0.2],
                 n_step_decoder=series_step,
                 default=True,
                 negative=True)
val_x, val_y, val_prev_y = DATA.validation()
test_x, test_y, test_prev_y = DATA.test()

feature_idx = 0
for feature in DATA.data_info:
    if not feature["active"]:
        continue
    if feature["name"] == "time-trans":
        val_x[feature_idx] = val_x[feature_idx] / 60 / 60 / dtime_thershold
        test_x[feature_idx] = test_x[feature_idx] / 60 / 60 / dtime_thershold
        DATA.train_x[feature_idx] = DATA.train_x[feature_idx] / 60 / 60 / dtime_thershold
    if feature["name"] == "geo-trans":
        val_x[feature_idx] = val_x[feature_idx] / dgeo_thershold
        test_x[feature_idx] = test_x[feature_idx] / dgeo_thershold
        DATA.train_x[feature_idx] = DATA.train_x[feature_idx] / dgeo_thershold
    feature_idx += 1

table = Texttable(0)
table.set_deco(Texttable.BORDER | Texttable.VLINES | Texttable.HEADER)
table.set_cols_dtype(['t', 'a'])
table.set_cols_align(['c', 'c'])
table.add_rows([["Train Info", "Value"],
                ["Running time", time.ctime()],
                ["Log directory", log_path]])
table_info = table.draw() + "\n"
table = Texttable(0)
table.set_deco(Texttable.BORDER | Texttable.VLINES | Texttable.HEADER)
table.set_cols_dtype(['a'] * 13)
table.set_cols_align(['c'] * 13)
table.add_rows([["Epochs", "Batch", "Learn", "Hidden", "Sample",
                 "Series", "time_s", "space_s", "RNNCell",
                 "Opt", "Display", "Dataset", "Reg"],
                [epochs, train_size, learning_rate, hidden_units, sample,
                 series_step, dtime_thershold, dgeo_thershold, rnn,
                 opt_method, display_step, dataset_name, reg]])
table_info = table_info + table.draw() + "\n"
print(table_info)

print("Start!")
# =========================================== MODEL START ===========================================
print("Start build model")

# ========= placeholder
input_x = []
for j, feature in enumerate(DATA.data_info):
    if not feature["active"]:
        continue
    if feature["format"] == "embed":
        tmp = tf.placeholder(dtype=tf.int32, shape=(None, n_steps_encoder, 1), name="x_{}".format(feature["name"]))
    elif feature["format"] == "interp":
        tmp = tf.placeholder(dtype=tf.float32, shape=(None, n_steps_encoder, 1), name="x_{}".format(feature["name"]))
    elif feature["format"] == "list":
        tmp = tf.placeholder(dtype=tf.float32, shape=(None, n_steps_encoder, feature["num"] + 1),
                             name="x_{}".format(feature["name"]))
    elif feature["format"] == "float":
        tmp = tf.placeholder(dtype=tf.float32, shape=(None, n_steps_encoder, 1), name="x_{}".format(feature["name"]))
    else:
        raise ValueError("10001")
    input_x.append(tmp)
ones_matrix = tf.placeholder(dtype=tf.float32, shape=(None, 1))
# prev input
input_pre_y = tf.placeholder(dtype=tf.int32, shape=(None, n_steps_decoder, 1), name="y_prev")
# output
y_label = tf.placeholder(dtype=tf.int32, shape=(None,), name="y_poi")
y_label_neg = tf.placeholder(dtype=tf.int32, shape=(None,), name="y_neg")
# ==================
e_y_pred = TMCAModel.TMCAModel(input_x, input_pre_y,
                               DATA.data_info, DATA.output_num, DATA.output_dim,
                               n_steps_encoder, n_steps_decoder,
                               n_hidden_encoder, n_hidden_decoder,
                               ones_matrix,
                               frag_mode=frag_mode)

# ==================
initializer = tf.random_normal_initializer(0.0, 0.1)
w_out = tf.get_variable("w_out", [DATA.output_num, DATA.output_dim], tf.float32, initializer)
b_out = tf.get_variable("b_out", [DATA.output_num], tf.float32, initializer)
with tf.name_scope("cost"):  # classification
    y_pred = tf.matmul(e_y_pred, w_out, transpose_b=True) + b_out  # 'cost/add:0'
    cost_list = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label, logits=y_pred)
    # cost_list_neg = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_label_neg, logits=y_pred)
    cost = tf.reduce_mean(cost_list)
# optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
# generate recommendetion list:
pred_top_10 = tf.nn.top_k(y_pred, max(topk_list))
# =========================================== MODEL END ==============================================
# summary
tf.summary.scalar('cost', cost)
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

# save the model
val_saver = tf.train.Saver(max_to_keep=max_to_keep)
test_saver = tf.train.Saver(max_to_keep=max_to_keep)
loss_value = []
step_value = []
recall_test_saver = []
recall_val_saver = []
max_val1_epoch = 0
max_val1_table = 0
max_val1_auc = 0
max_val1_loss = 0.
max_test1_epoch = 0
max_test1_table = 0
max_test1_auc = 0
max_test1_loss = 0.

epoch = 0
small_size = train_size
display_start_time = time.clock()

print("start run")
gpu_option = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))
writer = tf.summary.FileWriter(summary_path, sess.graph)
sess.run(init)
start_time = time.clock()

while epoch < epochs:
    batch_x, batch_y, batch_prev_y, batch_y_neg = DATA.next_batch(train_size)
    feed_dict = {}
    
    feature_idx = 0
    for feature in DATA.data_info:
        if feature["active"]:
            if feature["format"] == "list":
                feed_dict[input_x[feature_idx]] = hstack(batch_x[feature_idx],
                                                         feature["num"] + 1, n_steps_encoder)
            else:
                feed_dict[input_x[feature_idx]] = batch_x[feature_idx]
            feature_idx += 1
    feed_dict.update({input_pre_y: batch_prev_y,
                      y_label: batch_y,
                      y_label_neg: batch_y_neg,
                      ones_matrix: np.ones(shape=[train_size, 1], dtype=np.float32)})
    sess.run(optimizer, feed_dict)
    
    # =============== display time ================
    in_num = (epoch + 1) % display_step
    if (epoch + 1) % 20 == 0 and in_num != 0:
        per_run_time = (time.clock() - start_time) / in_num
        print("Epoch:{}/{}, ET: {}".
              format(in_num, display_step,
                     tools.print_time(0, per_run_time * (display_step - in_num))),
              end='\r')
        
    # ===================== display step =====================
    if in_num == 0:
        cost_value = 0.
        recall_ndcg_train = np.zeros(len(topk_list) * 2, dtype=float)
        for st, et in tools.range_small_dataset(DATA.train_y.shape[0], small_size):
            print("train: {}/{}".format(st, DATA.train_y.shape[0]), end='\r')
            feed_dict = {}

            feature_idx = 0
            for feature in DATA.data_info:
                if feature["active"]:
                    if feature["format"] == "list":
                        feed_dict[input_x[feature_idx]] = hstack(DATA.train_x[feature_idx][st: et],
                                                                 feature["num"] + 1, n_steps_encoder)
                    else:
                        feed_dict[input_x[feature_idx]] = DATA.train_x[feature_idx][st: et, :, :]
                    feature_idx += 1
            feed_dict.update({input_pre_y: DATA.train_prev_y[st: et, :, :],
                              y_label: DATA.train_y[st: et],
                              ones_matrix: np.ones(shape=[et - st, 1], dtype=np.float32)})
            cost_value += (et - st) * sess.run(cost, feed_dict)
        print("Epoch: {} , Loss: {:.6f}".format(epoch + 1, cost_value))
        loss_value.append(cost_value)
        step_value.append(epoch + 1)
        
        # ===================== validation =====================
        recall_ndcg_val = np.zeros(len(topk_list) * 2, dtype=float)
        for st, et in tools.range_small_dataset(val_y.shape[0], small_size):
            print("val: {}/{}".format(st, val_y.shape[0]), end='\r')
            feed_dict = {}
            
            feature_idx = 0
            for feature in DATA.data_info:
                if feature["active"]:
                    if feature["format"] == "list":
                        feed_dict[input_x[feature_idx]] = hstack(val_x[feature_idx][st: et],
                                                                 feature["num"] + 1, n_steps_encoder)
                    else:
                        feed_dict[input_x[feature_idx]] = val_x[feature_idx][st: et, :, :]
                    feature_idx += 1
            feed_dict.update({input_pre_y: val_prev_y[st: et, :, :],
                              ones_matrix: np.ones(shape=[et - st, 1], dtype=np.float32)})
            pred_val = sess.run(pred_top_10, feed_dict)
            recall_ndcg_val_s = np.r_[tools.add_recall_ndcg(val_y[st: et].reshape(-1), pred_val[1], topk_list)]
            recall_ndcg_val += recall_ndcg_val_s
            print("val: {0}/{1}: {2}/{3} ({4})".format(st, val_y.shape[0], recall_ndcg_val_s[0], et - st,
                                                       recall_ndcg_val_s[0] / (et - st)), end='\r')
        recall_val = [x / val_y.shape[0] for x in recall_ndcg_val[:len(topk_list)]]
        ndcg_val = [x / val_y.shape[0] for x in recall_ndcg_val[len(topk_list):]]
        recall_val_saver.append(recall_val)
        # ===================== Test ==========================
        recall_ndcg_test = np.zeros(len(topk_list) * 2, dtype=float)
        true_pos_test = None
        for st, et in tools.range_small_dataset(test_y.shape[0], small_size):
            print("test: {}/{}".format(st, test_y.shape[0]), end='\r')
            feed_dict = {}
            
            feature_idx = 0
            for feature in DATA.data_info:
                if feature["active"]:
                    if feature["format"] == "list":
                        feed_dict[input_x[feature_idx]] = hstack(test_x[feature_idx][st: et],
                                                                 feature["num"] + 1, n_steps_encoder)
                    else:
                        feed_dict[input_x[feature_idx]] = test_x[feature_idx][st: et, :, :]
                    feature_idx += 1
            feed_dict.update({input_pre_y: test_prev_y[st: et, :, :],
                              ones_matrix: np.ones(shape=[et - st, 1], dtype=np.float32)})
            pred_test = sess.run(pred_top_10, feed_dict)
            recall_ndcg_test_s = np.r_[
                tools.add_recall_ndcg(test_y[st: et].reshape(-1), pred_test[1], topk_list)]
            recall_ndcg_test += recall_ndcg_test_s
            print("test: {0}/{1}: {2}/{3} ({4})".format(st, test_y.shape[0], recall_ndcg_test_s[0], et - st,
                                                        recall_ndcg_test_s[0] / (et - st)), end='\r')
        recall_test = [x / test_y.shape[0] for x in recall_ndcg_test[:len(topk_list)]]
        ndcg_test = [x / test_y.shape[0] for x in recall_ndcg_test[len(topk_list):]]
        recall_test_saver.append(recall_test)
        
        # show
        table_ = tools.print_evaluation(topk_list, recall_val + recall_test, ndcg_val + ndcg_test)
        # save
        if recall_val[5] >= np.max(recall_val_saver, axis=0)[5]:
            max_val1_epoch = epoch + 1
            max_val1_table = table_
            max_val1_loss = cost_value
            val_saver.save(sess, val_path + method_name + '_' + str(epoch + 1) + '.ckpt')
        if recall_test[5] >= np.max(recall_test_saver, axis=0)[5]:
            max_test1_epoch = epoch + 1
            max_test1_table = table_
            max_test1_loss = cost_value
            test_saver.save(sess, test_path + method_name + '_' + str(epoch + 1) + '.ckpt')
        
        print("display runtime: {}".format(tools.print_time(display_start_time, time.clock())))
        display_start_time = time.clock()
        start_time = time.clock()
    epoch += 1

print("Optimization Finished!")
sess.close()
all_saver = np.c_[np.array(step_value), loss_value,
                  recall_val_saver, recall_test_saver]
table_header = ["epoch", "loss"]
for k in topk_list:
    table_header.append("val_recall_{}".format(k))
for k in topk_list:
    table_header.append("test_recall_{}".format(k))
pd.DataFrame(all_saver).to_csv(log_path + "result.csv", index=False, header=table_header)

print(table_info)
print("Epoch (max val@K): ", max_val1_epoch, "Loss: ", max_val1_loss)
print(max_val1_table)
print("Epoch (max test@K)", max_test1_epoch, "Loss: ", max_test1_loss)
print(max_test1_table)
with open(log_path + "Info.txt", "w") as f:
    f.write(table_info)
    f.write("\nEpoch (max val@K): {}, Loss: {:.6f}\n".format(max_val1_epoch, float(max_val1_loss)))
    f.write(max_val1_table)
    f.write("\nEpoch (max test@K): {}, Loss: {:.6f}\n".format(max_test1_epoch, float(max_test1_loss)))
    f.write(max_test1_table)
