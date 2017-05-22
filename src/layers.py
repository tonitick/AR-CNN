import tensorflow as tf
import numpy as np
from utils import *
import os

def get_weights(shape):
    # weights_initializer = tf.contrib.layers.xavier_initializer()    
    # W = tf.get_variable(name, shape, tf.float32, weights_initializer)
    initial = tf.truncated_normal(shape, stddev=0.001, dtype=tf.float32)
    W = tf.Variable(initial, dtype=tf.float32)

    return W

def get_bias(shape):
    # initial = tf.constant(0.0, shape=shape, dtype=tf.float32)
    initial = tf.truncated_normal(shape, stddev=0.001, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)

def conv_layer(conf, layer_name, fan_in, kernel_size, f_map, activation):
    with tf.variable_scope(layer_name):
        if conf.phase == 'train':
            with tf.variable_scope('W'):
                W_shape = [kernel_size, kernel_size, fan_in.get_shape()[-1].value, f_map]
                W = get_weights(W_shape)
                variable_summaries(W)
            
            with tf.variable_scope('conv'):
                conv = tf.nn.conv2d(fan_in, W, strides=[1,1,1,1], padding='SAME')
                variable_summaries(conv)

            with tf.variable_scope('b'):
                b = get_bias([f_map])
                variable_summaries(b)

            with tf.variable_scope('fan_out'):
                if activation == True:
                    fan_out = tf.nn.relu(tf.nn.bias_add(conv, b))
                else:
                    fan_out = tf.nn.bias_add(conv, b)
                variable_summaries(fan_out)

        else: # conf.phase == 'test'
            data_file = os.path.join(conf.param_path, 'quality=' + str(conf.quality), layer_name + '.npz')
            load_data = np.load(data_file)

            W = tf.Variable(load_data['W'])
            conv = tf.nn.conv2d(fan_in, W, strides=[1,1,1,1], padding='SAME')
            b = tf.Variable(load_data['b'])
            if activation == True:
                fan_out = tf.nn.relu(tf.nn.bias_add(conv, b))
            else:
                fan_out = tf.nn.bias_add(conv, b)

    return W, b, fan_out
