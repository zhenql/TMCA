# coding=utf-8
# ================================
# THIS FILE IS PART OF TMCA
# AttentionEncoder.py - The Encoder part of the TMCA model
#
# We modify some parts of attention_decoder in tensorflow package.
#
# Copyright (C) 2018 Ranzhen Li. All Rights Reserved
# ================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import rnn_cell_impl as core_rnn_cell_impl
from tensorflow.python.framework import dtypes
# from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


def attention_encoder(encoder_inputs,
                      attention_states,
                      cell,
                      output_size=None,
                      dtype=dtypes.float32,
                      scope=None):
    """
    RNN encoder with attention.
    In this context "attention" means that, during encoding, the RNN can look up
    information in the additional tensor "attention_states", which is constructed
    by transpose the dimensions of time steps and input features of the inputs,
    and it does this to focus on a few features of the input.

    Args:
        encoder_inputs: A list of 2D Tensors
            `n_feature x n_steps x [batch_size x feature_dim]`.
        attention_states: A list of 3D Tensor
            `n_feature x [batch_size x features_dim x n_steps]`.
        cell: rnn_cell.RNNCell defining the cell function and size.
            Now: only support BasicLSTMCell
        output_size: Size of the output vectors;
            if None, we use cell.output_size.
        dtype: The dtype to use for the RNN initial state
            default: tf.float32.
        scope: VariableScope for the created subgraph;
            default: "HierarchicalAttn_1".

    Returns:
        A tuple of the form (outputs, state, attns_context, attns_feature), where:
        outputs: A list of the encoder hidden states.
                Each element is a 2D Tensor of shape [batch_size x output_size].
        state: The state of encoder cell at the final time-step.
                It is a 2D Tensor of shape [batch_size x cell.state_size].
        attns_context: A list of the context attention weights.
                Each element is a 2D Tensor of shape [batch_size x all_feature_dims]
        attns_feature: A list of the feature attention weights.
                Each element is a 2D Tensor of shape [batch_size x n_feature]
    Raises:
        ValueError: when num_heads is not positive, there are no inputs, shapes
            of attention_states are not set, or input size cannot be inferred
            from the input.
    """
    # print("Use hierarchical attention scheme.")
    if not encoder_inputs:
        raise ValueError("Must provide at least 1 input to attention encoder.")
    if not attention_states:
        raise ValueError("Must provide at least 1 attention state to attention encoder.")
    if output_size is None:
        output_size = cell.output_size
    
    with variable_scope.variable_scope(scope or "HierarchicalAttn"):
        
        # get the batch_size of the encoder_input
        batch_size = array_ops.shape(encoder_inputs[0][0])[0]  # Needed for reshaping.
        n_feature = len(attention_states)
        n_steps = len(encoder_inputs[0])
        feature_dims = [attn_state.get_shape()[1] for attn_state in attention_states]

        # U_c * x(i,`,f) for i in range(feature["dim"]) (filter)
        U_micro = variable_scope.get_variable("AttnEncoderU_micro",
                                              [1, 1, n_steps, n_steps], dtype=dtype)
        U_macro = []
        b_macro1 = variable_scope.get_variable("AttnEncoderb_macro1",
                                              [1, 1, n_steps, 1], dtype=dtype)
        
        # make x features.
        x_micro_features = []  # n_feature x [batch_size, features_dim, 1, n_steps]
        x_macro_features = []  # n_feature x [batch_size, n_steps, 1, features_dim]
        for f, feature_dim in enumerate(feature_dims):
            
            # make micro features
            x1 = array_ops.reshape(attention_states[f], [-1, feature_dim, 1, n_steps])
            x_micro_features.append(nn_ops.conv2d(x1, U_micro, [1, 1, 1, 1], "SAME"))
            
            U_macro.append(variable_scope.get_variable("AttnEncoderU_micro_%d" % f,
                                                       [1, 1, feature_dim, n_steps], dtype=dtype))
            
            # make mcro features
            x2 = nn_ops.conv2d(x1, b_macro1, [1, 1, 1, 1], "SAME")
            x3 = array_ops.transpose(x2, [0, 3, 2, 1])
            x_macro_features.append(nn_ops.conv2d(x3, U_macro[f], [1, 1, 1, 1], "SAME"))
            
        v_micro = variable_scope.get_variable("AttnEncoderv_micro", [n_steps], dtype=dtype)
        v_macro = variable_scope.get_variable("AttnEncoderv_macro", [n_steps], dtype=dtype)
        
        initial_state_size = array_ops.stack([batch_size, output_size])
        initial_state = [array_ops.zeros(initial_state_size, dtype=dtype) for _ in xrange(2)]
        state = initial_state
        
        W_micro = variable_scope.get_variable("AttnEncoderW_micro", [2*output_size, n_steps], dtype=dtype)
        b_micro = variable_scope.get_variable("AttnEncoderb_micro", [n_steps], dtype=dtype)
        W_macro = variable_scope.get_variable("AttnEncoderW_macro", [2*output_size, n_steps], dtype=dtype)
        b_macro2 = variable_scope.get_variable("AttnEncoderb_macro2", [n_steps], dtype=dtype)

        def micro_attention(query, feature_idx, step):
            if nest.is_sequence(query):
                query_list = nest.flatten(query)
                query = array_ops.concat(query_list, 1)
            _tmp = math_ops.matmul(query, W_micro) + b_micro
            _tmp = array_ops.reshape(_tmp, [-1, 1, 1, n_steps])
            a_micro = math_ops.reduce_sum(v_micro * math_ops.tanh(x_micro_features[feature_idx] + _tmp), [2, 3])
            alpha_micro = nn_ops.softmax(a_micro, name="alpha_micro_%d_%d" % (step, feature_idx))
            return alpha_micro
        
        def macro_attention(query, step):
            if nest.is_sequence(query):
                query_list = nest.flatten(query)
                query = array_ops.concat(query_list, 1)
            _tmp = math_ops.matmul(query, W_macro) + b_macro2
            _tmp = array_ops.reshape(_tmp, [-1, 1, 1, n_steps])
            a_macro = [math_ops.reduce_sum(v_macro * math_ops.tanh(x_macro_features[f_idx] + _tmp), [2, 3])
                       for f_idx in xrange(n_feature)]
            a_macro = array_ops.concat(a_macro, 1)
            alpha_macro = nn_ops.softmax(a_macro, name="alpha_macro_%d" % step)
            return alpha_macro
    
        outputs = []
        attns_micro = []
        attns_macro = []

        for i in xrange(n_steps):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            
            with variable_scope.variable_scope("MicroAttn"):
                attn_micro = [micro_attention(state, f_idx, i) for f_idx in xrange(n_feature)]
                attns_micro.append(array_ops.concat(attn_micro, 1))
                x_micro = [encoder_inputs[f_idx][i]*attn_micro[f_idx] for f_idx in xrange(n_feature)]

            with variable_scope.variable_scope("MacroAttn"):
                attn_macro = macro_attention(state, i)
                attns_macro.append(attn_macro)
                attn_macro2 = array_ops.split(attn_macro, num_or_size_splits=n_feature, axis=1)
                
                x_macro = [x_micro[f_idx]*attn_macro2[f_idx] for f_idx in xrange(n_feature)]
                x = array_ops.concat(x_macro, 1)
            
            with variable_scope.variable_scope("EncoderOutput"):
                output, state = cell(x, state)
                outputs.append(output)

    return outputs, state, attns_micro, attns_macro
