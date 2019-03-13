""" some basic function
Xinhai Liu
Date: June 2018
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
import tensorflow.contrib.seq2seq as seq2seq

def _variable_on_cpu(name, shape, initializer, use_fp16=False):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device("/cpu:0"):
    dtype = tf.float16 if use_fp16 else tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var

def _variable_with_weight_decay(name, shape, stddev, wd, use_xavier=True):
  """Helper to create an initialized Variable with weight decay.

  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.

  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
    use_xavier: bool, whether to use xavier initializer

  Returns:
    Variable Tensor
  """
  if use_xavier:
    initializer = tf.contrib.layers.xavier_initializer()
  else:
    initializer = tf.truncated_normal_initializer(stddev=stddev)
  var = _variable_on_cpu(name, shape, initializer)
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=1,
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 1D convolution with non-linear operation.

  Args:
    inputs: 3-D tensor variable BxLxC
    num_output_channels: int
    kernel_size: int
    scope: string
    stride: int
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    assert(data_format=='NHWC' or data_format=='NCHW')
    if data_format == 'NHWC':
      num_in_channels = inputs.get_shape()[-1].value
    elif data_format=='NCHW':
      num_in_channels = inputs.get_shape()[1].value
    kernel_shape = [kernel_size,
                    num_in_channels, num_output_channels]
    kernel = _variable_with_weight_decay('weights',
                                         shape=kernel_shape,
                                         use_xavier=use_xavier,
                                         stddev=stddev,
                                         wd=weight_decay)
    outputs = tf.nn.conv1d(inputs, kernel,
                           stride=stride,
                           padding=padding,
                           data_format=data_format)
    biases = _variable_on_cpu('biases', [num_output_channels],
                              tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

    if bn:
      outputs = batch_norm_for_conv1d(outputs, is_training,
                                      bn_decay=bn_decay, scope='bn',
                                      data_format=data_format)

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def rnn_encoder(inputs,
        hidden_size,
        scope,
        activation_fn=tf.nn.relu,
        bn=False,
        bn_decay=None,
        is_training=None):
  """ RNN encoder with no-linear operation.
  Args:
    inputs: 4-D tensor variable BxNxTxD
    hidden_size: int
    scope: encoder
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable
  Return:
    Variable Tensor BxNxD
  """
  with tf.variable_scope(scope) as sc:
    batch_size = inputs.get_shape()[0].value
    npoint = inputs.get_shape()[1].value
    nstep = inputs.get_shape()[2].value
    in_size = inputs.get_shape()[3].value
    reshaped_inputs = tf.reshape(inputs, (-1, nstep, in_size))
    # cell = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
    cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
    # cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
    h0 = cell.zero_state(batch_size*npoint, np.float32)
    output, state = tf.nn.dynamic_rnn(cell, reshaped_inputs, initial_state=h0)
    outputs = tf.reshape(state.h, (-1, npoint, hidden_size))

    if bn:
      outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs

def seq2seq_without_attention(inputs,
        hidden_size,
        scope,
        activation_fn=tf.nn.relu,
        bn=False,
        bn_decay=None,
        is_training=None):
    """ sequence model without attention.
    Args:
      inputs: 4-D tensor variable BxNxTxD
      hidden_size: int
      scope: encoder
      activation_fn: function
      bn: bool, whether to use batch norm
      bn_decay: float or float tensor variable in [0,1]
      is_training: bool Tensor variable
    Return:
      Variable Tensor BxNxD
    """
    with tf.variable_scope(scope) as sc:
        batch_size = inputs.get_shape()[0].value
        npoint = inputs.get_shape()[1].value
        nstep = inputs.get_shape()[2].value
        in_size = inputs.get_shape()[3].value
        reshaped_inputs = tf.reshape(inputs, (-1, nstep, in_size))

        with tf.variable_scope('encoder'):
            # build encoder
            encoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, reshaped_inputs,
                                                               sequence_length=tf.fill([batch_size * npoint], 4),
                                                               dtype=tf.float32, time_major=False)
        with tf.variable_scope('decoder'):
            decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            decoder_inputs = tf.reshape(encoder_state.h, [batch_size * npoint, 1, hidden_size])

            # Helper to feed inputs for training: read inputs from dense ground truth vectors
            train_helper = seq2seq.TrainingHelper(inputs=decoder_inputs, sequence_length=tf.fill([batch_size * npoint], 1),
                                                  time_major=False)
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size * npoint, dtype=tf.float32)
            train_decoder = seq2seq.BasicDecoder(cell=decoder_cell, helper=train_helper,
                                                 initial_state=decoder_initial_state, output_layer=None)
            decoder_outputs_train, decoder_last_state_train, decoder_outputs_length_train = seq2seq.dynamic_decode(
                decoder=train_decoder, output_time_major=False, impute_finished=True)
        outputs = tf.reshape(decoder_last_state_train.c, (-1, npoint, hidden_size))
        if bn:
            outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
            outputs = activation_fn(outputs)
        return outputs

def seq2seq_with_attention(inputs,
        hidden_size,
        scope,
        activation_fn=tf.nn.relu,
        bn=False,
        bn_decay=None,
        is_training=None):
    """ sequence model with attention.
       Args:
         inputs: 4-D tensor variable BxNxTxD
         hidden_size: int
         scope: encoder
         activation_fn: function
         bn: bool, whether to use batch norm
         bn_decay: float or float tensor variable in [0,1]
         is_training: bool Tensor variable
       Return:
         Variable Tensor BxNxD
       """
    with tf.variable_scope(scope) as sc:
        batch_size = inputs.get_shape()[0].value
        npoint = inputs.get_shape()[1].value
        nstep = inputs.get_shape()[2].value
        in_size = inputs.get_shape()[3].value
        reshaped_inputs = tf.reshape(inputs, (-1, nstep, in_size))

        with tf.variable_scope('encoder'):
            #build encoder
            encoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, reshaped_inputs,
                                                               sequence_length=tf.fill([batch_size*npoint], 4),
                                                               dtype=tf.float32, time_major=False)
        with tf.variable_scope('decoder'):
            #build decoder
            decoder_cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            decoder_inputs = tf.reshape(encoder_state.h, [batch_size*npoint, 1, hidden_size])

            # building attention mechanism: default Bahdanau
            # 'Bahdanau' style attention: https://arxiv.org/abs/1409.0473
            attention_mechanism = seq2seq.BahdanauAttention(num_units=hidden_size, memory=encoder_outputs)
            # 'Luong' style attention: https://arxiv.org/abs/1508.04025
            # attention_mechanism = seq2seq.LuongAttention(num_units=hidden_size, memory=encoder_outputs)

            # AttentionWrapper wraps RNNCell with the attention_mechanism
            decoder_cell = seq2seq.AttentionWrapper(cell=decoder_cell, attention_mechanism=attention_mechanism,
                                                              attention_layer_size=hidden_size)

            # Helper to feed inputs for training: read inputs from dense ground truth vectors
            train_helper = seq2seq.TrainingHelper(inputs=decoder_inputs, sequence_length=tf.fill([batch_size*npoint], 1),
                                                  time_major=False)
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size*npoint, dtype=tf.float32)
            train_decoder = seq2seq.BasicDecoder(cell=decoder_cell, helper=train_helper, initial_state=decoder_initial_state, output_layer=None)
            decoder_outputs_train, decoder_last_state_train, decoder_outputs_length_train = seq2seq.dynamic_decode(
                decoder=train_decoder, output_time_major=False, impute_finished=True)

        outputs = tf.reshape(decoder_last_state_train[0].h, (-1, npoint, hidden_size))
        if bn:
          outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

        if activation_fn is not None:
          outputs = activation_fn(outputs)
        return outputs

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope,
           stride=[1, 1],
           padding='SAME',
           data_format='NHWC',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=None,
           activation_fn=tf.nn.relu,
           bn=False,
           bn_decay=None,
           is_training=None):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    data_format: 'NHWC' or 'NCHW'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      assert(data_format=='NHWC' or data_format=='NCHW')
      if data_format == 'NHWC':
        num_in_channels = inputs.get_shape()[-1].value
      elif data_format=='NCHW':
        num_in_channels = inputs.get_shape()[1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_in_channels, num_output_channels]
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride
      outputs = tf.nn.conv2d(inputs, kernel,
                             [1, stride_h, stride_w, 1],
                             padding=padding,
                             data_format=data_format)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases, data_format=data_format)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn',
                                        data_format=data_format)

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs


def conv2d_transpose(inputs,
                     num_output_channels,
                     kernel_size,
                     scope,
                     stride=[1, 1],
                     padding='SAME',
                     use_xavier=True,
                     stddev=1e-3,
                     weight_decay=None,
                     activation_fn=tf.nn.relu,
                     bn=False,
                     bn_decay=None,
                     is_training=None):
  """ 2D convolution transpose with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor

  Note: conv2d(conv2d_transpose(a, num_out, ksize, stride), a.shape[-1], ksize, stride) == a
  """
  with tf.variable_scope(scope) as sc:
      kernel_h, kernel_w = kernel_size
      num_in_channels = inputs.get_shape()[-1].value
      kernel_shape = [kernel_h, kernel_w,
                      num_output_channels, num_in_channels] # reversed to conv2d
      kernel = _variable_with_weight_decay('weights',
                                           shape=kernel_shape,
                                           use_xavier=use_xavier,
                                           stddev=stddev,
                                           wd=weight_decay)
      stride_h, stride_w = stride

      # from slim.convolution2d_transpose
      def get_deconv_dim(dim_size, stride_size, kernel_size, padding):
          dim_size *= stride_size

          if padding == 'VALID' and dim_size is not None:
            dim_size += max(kernel_size - stride_size, 0)
          return dim_size

      # caculate output shape
      batch_size = inputs.get_shape()[0].value
      height = inputs.get_shape()[1].value
      width = inputs.get_shape()[2].value
      out_height = get_deconv_dim(height, stride_h, kernel_h, padding)
      out_width = get_deconv_dim(width, stride_w, kernel_w, padding)
      output_shape = [batch_size, out_height, out_width, num_output_channels]

      outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape,
                             [1, stride_h, stride_w, 1],
                             padding=padding)
      biases = _variable_on_cpu('biases', [num_output_channels],
                                tf.constant_initializer(0.0))
      outputs = tf.nn.bias_add(outputs, biases)

      if bn:
        outputs = batch_norm_for_conv2d(outputs, is_training,
                                        bn_decay=bn_decay, scope='bn')

      if activation_fn is not None:
        outputs = activation_fn(outputs)
      return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=None,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    is_training=None):
  """ Fully connected layer with non-linear operation.

  Args:
    inputs: 2-D tensor BxN
    num_outputs: int

  Returns:
    Variable tensor of size B x num_outputs.
  """
  with tf.variable_scope(scope) as sc:
    num_input_units = inputs.get_shape()[-1].value
    weights = _variable_with_weight_decay('weights',
                                          shape=[num_input_units, num_outputs],
                                          use_xavier=use_xavier,
                                          stddev=stddev,
                                          wd=weight_decay)
    outputs = tf.matmul(inputs, weights)
    biases = _variable_on_cpu('biases', [num_outputs],
                             tf.constant_initializer(0.0))
    outputs = tf.nn.bias_add(outputs, biases)

    if bn:
      outputs = batch_norm_for_fc(outputs, is_training, bn_decay, 'bn')

    if activation_fn is not None:
      outputs = activation_fn(outputs)
    return outputs


def max_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D max pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.max_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs

def avg_pool2d(inputs,
               kernel_size,
               scope,
               stride=[2, 2],
               padding='VALID'):
  """ 2D avg pooling.

  Args:
    inputs: 4-D tensor BxHxWxC
    kernel_size: a list of 2 ints
    stride: a list of 2 ints

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    outputs = tf.nn.avg_pool(inputs,
                             ksize=[1, kernel_h, kernel_w, 1],
                             strides=[1, stride_h, stride_w, 1],
                             padding=padding,
                             name=sc.name)
    return outputs


def max_pool3d(inputs,
               kernel_size,
               scope,
               stride=[2, 2, 2],
               padding='VALID'):
  """ 3D max pooling.

  Args:
    inputs: 5-D tensor BxDxHxWxC
    kernel_size: a list of 3 ints
    stride: a list of 3 ints

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope) as sc:
    kernel_d, kernel_h, kernel_w = kernel_size
    stride_d, stride_h, stride_w = stride
    outputs = tf.nn.max_pool3d(inputs,
                               ksize=[1, kernel_d, kernel_h, kernel_w, 1],
                               strides=[1, stride_d, stride_h, stride_w, 1],
                               padding=padding,
                               name=sc.name)
    return outputs


def batch_norm_template(inputs, is_training, scope, moments_dims_unused, bn_decay, data_format='NHWC'):
  """ Batch normalization on convolutional maps and beyond...
  Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow

  Args:
      inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
      is_training:   boolean tf.Varialbe, true indicates training phase
      scope:         string, variable scope
      moments_dims:  a list of ints, indicating dimensions for moments calculation
      bn_decay:      float or float tensor variable, controling moving average weight
      data_format:   'NHWC' or 'NCHW'
  Return:
      normed:        batch-normalized maps
  """
  bn_decay = bn_decay if bn_decay is not None else 0.9
  return tf.contrib.layers.batch_norm(inputs,
                                      center=True, scale=True,
                                      is_training=is_training, decay=bn_decay,updates_collections=None,
                                      scope=scope,
                                      data_format=data_format)


def batch_norm_for_fc(inputs, is_training, bn_decay, scope):
  """ Batch normalization on FC data.

  Args:
      inputs:      Tensor, 2D BxC input
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,], bn_decay)

def batch_norm_for_conv1d(inputs, is_training, bn_decay, scope, data_format):
  """ Batch normalization on 1D convolutional maps.

  Args:
      inputs:      Tensor, 3D BLC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      data_format: 'NHWC' or 'NCHW'
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1], bn_decay, data_format)

def batch_norm_for_conv2d(inputs, is_training, bn_decay, scope, data_format):
  """ Batch normalization on 2D convolutional maps.

  Args:
      inputs:      Tensor, 4D BHWC input maps
      is_training: boolean tf.Varialbe, true indicates training phase
      bn_decay:    float or float tensor variable, controling moving average weight
      scope:       string, variable scope
      data_format: 'NHWC' or 'NCHW'
  Return:
      normed:      batch-normalized maps
  """
  return batch_norm_template(inputs, is_training, scope, [0,1,2], bn_decay, data_format)


def dropout(inputs,
            is_training,
            scope,
            keep_prob=0.5,
            noise_shape=None):
  """ Dropout layer.

  Args:
    inputs: tensor
    is_training: boolean tf.Variable
    scope: string
    keep_prob: float in [0,1]
    noise_shape: list of ints

  Returns:
    tensor variable
  """
  with tf.variable_scope(scope) as sc:
    outputs = tf.cond(is_training,
                      lambda: tf.nn.dropout(inputs, keep_prob, noise_shape),
                      lambda: inputs)
    return outputs
