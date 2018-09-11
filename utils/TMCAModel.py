# coding=utf-8
# ================================
# THIS FILE IS PART OF TMCA
# TMCAModel.py - The model part of the TMCA model
#
# Copyright (C) 2018 Ranzhen Li. All Rights Reserved
# ================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl as rnn_cell
from utils import AttentionEncoder
from utils import AttentionDecoder
from utils.RichContext import get_encoder_input, get_decoder_input


def TMCAModel(input_x, input_pre_y,
              data_info, n_output_class, n_output_embed,
              n_steps_encoder, n_steps_decoder,
              n_hidden_encoder, n_hidden_decoder,
              ones_matrix,
              frag_mode=[True, True]):
    """
    Complete TMCA model.
    
    :param input_x: Encoder input. Tensors.
        n_features x [n_batch_size x n_steps_encoder x n_feature_dim]
    :param input_pre_y: Decoder input. Tensor.
        [n_batch_size x n_steps_decoder x 1]
    :param data_info:  Data information.
        A list of dicts like
        [{"name": "user", "format": "embed", "num": 22209, "dim": 60, "active": true,
        "default": false}]
    :param n_output_class: The number of POIs. Int.
    :param n_output_embed: The dimensions of output POI's hidden vector. Int
    :param n_steps_encoder: The number of encoder steps. Int.
    :param n_steps_decoder: The number of decoder steps. Int.
        In our paper, n_steps_decoder = n_steps_encoder + 1.
    :param n_hidden_encoder: The dimensions of encoder LSTM cell states. Int.
    :param n_hidden_decoder: The dimensions of decoder LSTM cell states. Int.
    :param ones_matrix: Auxiliary vector. Tensor.
        [n_batch_size x 1]
    :param frag_mode: [BOOL, BOOL]
    
    :return e_y_pred: The hidden vector for next POI (prediction)
        [n_batch_size x n_output_embed]
    """
    
    # ==================== Rich Context Incorporation ========================
    initializer = tf.random_normal_initializer(0, 0.01)
    
    encoder_input, embeddings_set = get_encoder_input(input_x, data_info,
                                                      n_steps_encoder, ones_matrix,
                                                      initializer)
    
    decoder_input, w_y_prev = get_decoder_input(input_pre_y, n_steps_decoder,
                                                n_output_class, n_output_embed,
                                                initializer)
    embeddings_set.append(w_y_prev)
    
    # ====================== Encoder and Decoder ==============================
    if not (frag_mode[0] and frag_mode[1]):
        
        batch_size = tf.shape(encoder_input[0][0])[0]
        initial_state_size = tf.stack([batch_size, n_hidden_encoder])
        initial_state = [tf.zeros(initial_state_size, dtype=tf.float32) for _ in range(2)]
        
    # ============================= Encoder ===================================
    with tf.variable_scope('Encoder'):
        encoder_cell = rnn_cell.BasicLSTMCell(n_hidden_encoder, forget_bias=0.0)
        
        if frag_mode[0]:
            # Multi-context attention
            print("We will use multi-level context attention for encoder.")
            encoder_attention_states = [tf.transpose(inp, [1, 2, 0]) for inp in encoder_input]
            encoder_outputs, encoder_state, context_attns, feature_attns = \
                AttentionEncoder.attention_encoder(encoder_input, encoder_attention_states, encoder_cell)
        
        else:
            # Not use multi-context attention
            # Mean attention for encoder input
            print("We will use mean attention for encoder.")
            n_feature = len(encoder_input)
            alpha_f = 1 / n_feature
            x_index = 0
            for feature_idx, feature in enumerate(data_info):
                if feature["active"]:
                    if feature["format"] != "float":
                        alpha_c = 1 / feature["dim"] * alpha_f
                        encoder_input[x_index] = [inp_ * alpha_c
                                                  for inp_ in encoder_input[x_index]]
                    x_index += 1
            encoder_input = [tf.concat([inp[si] for inp in encoder_input], 1)
                             for si in range(n_steps_encoder)]
            
            # encoder
            encoder_outputs = []
            state_ = initial_state
            for step_idx, inp_ in enumerate(encoder_input):
                if step_idx > 0:
                    tf.get_variable_scope().reuse_variables()
                outp_, state_ = encoder_cell(inp_, state_)
                encoder_outputs.append(outp_)
                
    # ============================= Decoder ======================================
    with tf.variable_scope('Decoder'):

        decoder_cell = rnn_cell.BasicLSTMCell(n_hidden_decoder, forget_bias=0.0)
        
        if frag_mode[1]:
            
            # Temporal attention
            decoder_attention_states = tf.concat([tf.reshape(h_, [-1, 1, encoder_cell.output_size])
                                                  for h_ in encoder_outputs], 1)
            print("We will use temporal attention for decoder.")
            decoder_outputs, decoder_state, temporal_attns = \
                AttentionDecoder.attention_decoder(decoder_input, decoder_attention_states, decoder_cell,
                                                   output_size=n_output_embed)
            
        else:
            
            print("We will use mean attention for decoder.")
            # Not use temporal attention
            # Mean attention for decoder input
            H = tf.concat([tf.reshape(h_, [-1, n_hidden_encoder, 1]) for h_ in encoder_outputs], 2)
            h_tilde = tf.reduce_mean(H, 2)
            
            # decoder
            decoder_outputs = []
            state_ = initial_state
            for step_idx, inp_ in enumerate(decoder_input):
                if step_idx > 0:
                    tf.get_variable_scope().reuse_variables()
                inp = tf.concat([inp_, h_tilde], 1)
                outp_, state_ = decoder_cell(inp, state_)
                decoder_outputs.append(outp_)
                
    e_y_pred = decoder_outputs[-1]  # shape: (batch_size, embed_dim)
    # ===========================================================================
    
    return e_y_pred
