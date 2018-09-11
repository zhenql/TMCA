# coding=utf-8
# ================================
# THIS FILE IS PART OF TMCA
# RichContext.py - The Embedding part of the TMCA model
# - get_decoder_input: prepare decoder input.
# - get_encoder_input: prepare encoder input.
#       Incorporate heterogeneous contextual factors in a unified way.
#
# Copyright (C) 2018 Ranzhen Li. All Rights Reserved
# ================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def get_decoder_input(input_pre_y, n_steps_decoder, n_output_class, n_output_embed, initializer):
    """
    Prepare decoder inputs. (embedding)

    :param input_pre_y: Decoder input.
        [n_batch_size x n_steps_decoder x 1]
    :param n_steps_decoder: The number of decoder steps.
    :param n_output_class: The number of POIs. Int.
    :param n_output_embed: The dimensions of output POI's hidden vector. Int
    :param initializer: variable intializer.

    :return decoder_input, embeddings_prev_y
            decoder_input:
                n_feature x n_steps x [n_batch_size x n_feature_dim]
    """
    with tf.name_scope("input_prev_y"):
        input_pre_y = tf.transpose(input_pre_y, [1, 0, 2])
        input_pre_y = tf.reshape(input_pre_y, [-1, 1])
        input_pre_y = tf.split(input_pre_y, n_steps_decoder, 0)
    with tf.name_scope("embed_prev_y"):
        embeddings_prev_y = tf.get_variable("w_prev_y", [n_output_class, n_output_embed],
                                            initializer=initializer)
        decoder_input = [tf.reshape(tf.nn.embedding_lookup(embeddings_prev_y, input_pre_y[k]),
                                    [-1, n_output_embed]) for k in range(n_steps_decoder)]
        return decoder_input, embeddings_prev_y


def get_encoder_input(input_x, data_info,
                      n_steps_encoder, ones_matrix, initializer):
    """
    Prepare encoder inputs. (embedding)
    
    :param input_x: Encoder input.
        n_features x [n_batch_size x n_steps_encoder x n_feature_dim]
    :param data_info:  Data information.
        A list of dicts like
        [{"name": "user", "format": "embed", "num": 22209, "dim": 60, "active": true, "default": false}]
    :param n_steps_encoder: The number of encoder steps.
    :param ones_matrix: Auxiliary vector.
        [n_batch_size x 1]
    :param initializer: variable intializer.
    :return encoder_input, embedding_set
            encoder_input:
                n_feature x n_steps x [n_batch_size x n_feature_dim]
    """
    # =========================== Rich Context Incorporation ======================================
    # =========================== Reshape
    # transpose (batch_size, n_step_encoder, x_num) => ( n_step_encoder, batch_size, x_num)
    # reshape => (n_step_encoder * batch_size, x_num)
    # split => n_step_encoder * (batch_size, x_num)
    input_x_reshape = input_x.copy()
    x_index = 0
    for j, feature in enumerate(data_info):
        if not feature["active"]:
            continue
        with tf.name_scope("input_" + feature["name"]):
            input_x_reshape[x_index] = tf.transpose(input_x_reshape[x_index], [1, 0, 2])
            if feature["format"] == "list":
                input_x_reshape[x_index] = tf.reshape(input_x_reshape[x_index],
                                                      [-1, feature["num"]+feature["default"]])
            else:
                input_x_reshape[x_index] = tf.reshape(input_x_reshape[x_index], [-1, 1])
            input_x_reshape[x_index] = tf.split(input_x_reshape[x_index], n_steps_encoder, 0)
        x_index += 1
    # =========================== Embeding: n_features * n_step_encoder * batch_size * dim
    input_x_embed = []
    embeddings_set = []
    x_index = 0
    for j, feature in enumerate(data_info):
        if not feature["active"]:
            continue
        with tf.name_scope("embed_" + feature["name"]):
            if feature["format"] == "list":
                w_tmp = tf.get_variable("w_{}".format(feature["name"]),
                                        [feature["num"]+feature["default"], feature["dim"]],
                                        initializer=initializer)
                embeddings_set.append(w_tmp)
                tmp = [tf.matmul(x_, w_tmp)/tf.reduce_sum(x_) for x_ in input_x_reshape[x_index]]
            elif feature["format"] == "embed":
                w_tmp = tf.get_variable("w_{}".format(feature["name"]),
                                        [feature["num"] + feature["default"], feature["dim"]],
                                        initializer=initializer)
                embeddings_set.append(w_tmp)
                tmp = [tf.reshape(tf.nn.embedding_lookup(w_tmp, x_), [-1, feature["dim"]])
                       for x_ in input_x_reshape[x_index]]
            elif feature["format"] == "interp":
                w_tmp = [tf.get_variable("w_{}_{}".format(feature["name"], idx),
                                         [1, feature["dim"]],
                                         initializer=initializer) for idx in range(3)]
                embeddings_set.append(w_tmp)
                tmp = [tf.where(tf.less_equal(tf.reshape(x_, [-1, ]), 1),
                                tf.matmul(x_, w_tmp[0]) + tf.matmul(1 - x_, w_tmp[1]),
                                tf.matmul(ones_matrix, w_tmp[2])) for x_ in input_x_reshape[x_index]]
            else:  # float and int
                # w_tmp = tf.get_variable("w_{}".format(feature["name"]), [1, feature["dim"]],
                #                         initializer=initializer)
                # embeddings_set.append(w_tmp)
                # tmp = [tf.matmul(x_, w_tmp) for x_ in input_x_reshape[x_index]]
                tmp = input_x_reshape[x_index]
        input_x_embed.append(tmp)
        x_index += 1
    # n_features x n_step_encoder x [batch_size x dim]
    # tf.transpose(input_x_embed, [1, 2, 3, 0])
    return input_x_embed, embeddings_set
