# coding=utf-8
# ================================
# THIS FILE IS PART OF TMCA
# AttentionDecoder.py - The Decoder part of the TMCA model
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
from tensorflow.python.ops import rnn_cell_impl as core_rnn_cell_impl
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

from tensorflow.python.ops import rnn_cell_impl

# TODO(ebrevdo): Remove once _linear is fully deprecated.
Linear = rnn_cell_impl._Linear  # pylint: disable=protected-access,invalid-name


def attention_decoder(decoder_inputs,
                      attention_states,
                      cell,
                      output_size=None,
                      dtype=None,
                      scope=None):
    """
    
    RNN decoder with attention for the sequence-to-sequence model.
    In this context "attention" means that, during decoding, the RNN can look up
    information in the additional tensor attention_states, and it does this by
    focusing on a few entries from the tensor.

    Args:
        decoder_inputs: A list of 2D Tensors [batch_size x input_size].
        attention_states: 3D Tensor [batch_size x attn_length x attn_size].
        cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
        output_size: Size of the output vectors; if None, we use cell.output_size.
        dtype: The dtype to use for the RNN initial state (default: tf.float32).
        scope: VariableScope for the created subgraph; default: "attention_decoder".

    Returns:
    A tuple of the form (outputs, state, temproal_attns), where:
        outputs: A list of the same length as decoder_inputs of 2D Tensors of
            shape [batch_size x output_size]. These represent the generated outputs.
            Output i is computed from input i (which is either the i-th element
            of decoder_inputs or loop_function(output {i-1}, i)) as follows.
            First, we run the cell on a combination of the input and previous
            attention masks:
                cell_output, new_state = cell(linear(input, prev_attn), prev_state).
            Then, we calculate new attention masks:
                new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
            and then we calculate the output:
                output = linear(cell_output, new_attn).
        state: The state of each decoder cell the final time-step.
            It is a 2D Tensor of shape [batch_size x cell.state_size].
        attns: The temporal attention.

    Raises:
        ValueError: when num_heads is not positive, there are no inputs, shapes
            of attention_states are not set, or input size cannot be inferred
            from the input.
    """
    if not decoder_inputs:
        raise ValueError("Must provide at least 1 input to attention decoder.")
    if output_size is None:
        output_size = cell.output_size
        
    # ==================================scope=================================================
    with variable_scope.variable_scope(scope or "TemporalAttn", dtype=dtype) as scope:
        
        dtype = scope.dtype
        batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
        attn_length = attention_states.get_shape()[1].value
        attn_size = attention_states.get_shape()[2].value
        
        # To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
        hidden = array_ops.reshape(attention_states, [-1, attn_length, 1, attn_size])
        # U_d * h_i for i in range(T) (filter)
        u = variable_scope.get_variable("AttnDecoderU", [1, 1, attn_size, attn_size], dtype=dtype)
        hidden_features = nn_ops.conv2d(hidden, u, [1, 1, 1, 1], "SAME")
        
        v = variable_scope.get_variable("AttnDecoderV", [attn_size], dtype=dtype)
        
        # how to get the initial_state
        initial_state_size = array_ops.stack([batch_size, cell.output_size])
        initial_state = [array_ops.zeros(initial_state_size, dtype=dtype) for _ in xrange(2)]
        state = initial_state
        
        w = variable_scope.get_variable("AttnDecoderW", [2*cell.output_size, attn_size], dtype=dtype)
        b = variable_scope.get_variable("AttnDecoderb", [attn_size], dtype=dtype)
        
        # beta_scalar = variable_scope.get_variable("BetaScalar", [attn_length])
        
        def attention(query, step):
            """
            Put attention masks on hidden using hidden_features and query.
            """
            
            if nest.is_sequence(query):  # If the query is a tuple, flatten it.
                query_list = nest.flatten(query)
                query = array_ops.concat(query_list, 1)
            _tmp = math_ops.matmul(query, w) + b
            _tmp = array_ops.reshape(_tmp, [-1, 1, 1, attn_size])
            # Attention mask is a softmax of v^T * tanh(...).
            s = math_ops.reduce_sum(v * math_ops.tanh(hidden_features + _tmp), [2, 3])
            # beta = math_ops.multiply(nn_ops.softmax(s, name="beta_%d" % step), beta_scalar)
            beta = nn_ops.softmax(s, name="beta_%d" % step)
            # Now calculate the attention-weighted vector d.
            
            hidden_attn = math_ops.reduce_sum(array_ops.reshape(beta, [-1, attn_length, 1, 1]) * hidden,
                                              [1, 2])
            return hidden_attn, beta

        outputs = []
        attns = []
        with variable_scope.variable_scope("Attn"):
            h_t, attn_t = attention(state, 0)
            attns.append(attn_t)
        # =============================recurrent===========================
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
                
            # LSTM_d([\tilde{\mathbf{h}}_{t}; \mathbf{y}_t], \hat{\mathbf{y}}_{t}, \mathbf{s}^d_{t})
            with variable_scope.variable_scope("DecoderOutput"):
                x = tf.concat([inp, h_t], 1)
                cell_output, state = cell(x, state)
                outputs.append(cell_output)

            with variable_scope.variable_scope("Attn"):
                h_t, attn_t = attention(state, i+1)
                attns.append(attn_t)
            
            with variable_scope.variable_scope("AttnDecoderOutput"):
                inputs = tf.concat([cell_output, h_t], 1)
                output = Linear(inputs, output_size, True)(inputs)
                outputs.append(output)
            
    return outputs, state, attns

