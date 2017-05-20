import tensorflow as tf
from layers import *

class ARCNN(object):
    def __init__(self, conf):
        self.compres = tf.placeholder(tf.float32, [None, conf.img_height, conf.img_width, conf.channel])
        self.truths = tf.placeholder(tf.float32, [None, conf.img_height, conf.img_width, conf.channel])        

        # feature extraction
        with tf.variable_scope("F_1"):
            F_1 = conv_layer(self.compres, 9, 64, True)
        
        # feature enhancement
        with tf.variable_scope("F_2"):
            F_2 = conv_layer(F_1, 7, 32, True)

        # mapping
        with tf.variable_scope("F_3"):
            F_3 = conv_layer(F_2, 1, 16, True)

        # reconstruction
        with tf.variable_scope("F_4"):
            F_4 = conv_layer(F_3, 5, conf.channel, False)

        shift_height = (conf.img_height - conf.valid_height) / 2
        shift_width = (conf.img_width - conf.valid_width) / 2
        mid_reconstruct = tf.strided_slice(F_4, [0, shift_height, shift_width, 0], \
                [conf.batch_size, shift_height + conf.valid_height, shift_width + conf.valid_width, conf.channel])
        mid_truth = tf.strided_slice(self.truths, [0, shift_height, shift_width, 0], \
                [conf.batch_size, shift_height + conf.valid_height, shift_width + conf.valid_width, conf.channel])
        
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(mid_reconstruct - mid_truth))
            tf.summary.scalar('loss', self.loss)
        
