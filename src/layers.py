import tensorflow as tf
import numpy as np
from utils import *

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

def load_weights(shape):
    initial = tf.truncated_normal(shape, stddev=0.001, dtype=tf.float32)
    W = tf.Variable(initial, dtype=tf.float32)

    return W

def load_bias(shape):
    initial = tf.truncated_normal(shape, stddev=0.001, dtype=tf.float32)
    return tf.Variable(initial, dtype=tf.float32)


def conv_layer_random(fan_in, kernel_size, f_map, activation):
    with tf.variable_scope("W"):
        W_shape = [kernel_size, kernel_size, fan_in.get_shape()[-1].value, f_map]
        W = get_weights(W_shape)
        variable_summaries(W)
    
    with tf.variable_scope("conv"):
        conv = tf.nn.conv2d(fan_in, W, strides=[1,1,1,1], padding='SAME')
        variable_summaries(conv)

    with tf.variable_scope("b"):
        b = get_bias(shape=[f_map])
        variable_summaries(b)

    with tf.variable_scope("fan_out"):
        if activation == True:
            fan_out = tf.nn.relu(tf.nn.bias_add(conv, b))
        else:
            fan_out = tf.nn.bias_add(conv, b)
        variable_summaries(fan_out)

    return fan_out

def conv_layer_load(fan_in, kernel_size, f_map, activation):
    with tf.variable_scope("W"):
        W_shape = [kernel_size, kernel_size, fan_in.get_shape()[-1].value, f_map]
        W = get_weights(W_shape)
        variable_summaries(W)
    
    with tf.variable_scope("conv"):
        conv = tf.nn.conv2d(fan_in, W, strides=[1,1,1,1], padding='SAME')
        variable_summaries(conv)

    with tf.variable_scope("b"):
        b = get_bias(shape=[f_map])
        variable_summaries(b)

    with tf.variable_scope("fan_out"):
        if activation == True:
            fan_out = tf.nn.relu(tf.nn.bias_add(conv, b))
        else:
            fan_out = tf.nn.bias_add(conv, b)
        variable_summaries(fan_out)

    return fan_out


   



