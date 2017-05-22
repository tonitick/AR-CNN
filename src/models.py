import tensorflow as tf
from layers import *
import os

class ARCNN(object):
    def __init__(self, conf, truths, compres):
        self.conf = conf
        self.truths = truths
        self.compres = compres

        # feature extraction
        self.W_1, self.b_1, self.F_1 = conv_layer(conf, 'feature_extraction', self.compres, 9, 64, True)
        
        # feature enhancement
        self.W_2, self.b_2, self.F_2 = conv_layer(conf, 'feature_enhancement', self.F_1, 7, 32, True)

        # mapping
        self.W_3, self.b_3, self.F_3 = conv_layer(conf, 'mapping', self.F_2, 1, 16, True)

        # reconstruction
        self.W_4, self.b_4, self.F_4 = conv_layer(conf, 'reconstruction', self.F_3, 5, conf.channel, False)

        shift_height = (conf.img_height - conf.valid_height) / 2
        shift_width = (conf.img_width - conf.valid_width) / 2
        mid_compres = tf.strided_slice(self.compres, [0, shift_height, shift_width, 0], \
                [conf.batch_size, shift_height + conf.valid_height, shift_width + conf.valid_width, conf.channel])
        mid_reconstruct = tf.strided_slice(self.F_4, [0, shift_height, shift_width, 0], \
                [conf.batch_size, shift_height + conf.valid_height, shift_width + conf.valid_width, conf.channel])
        mid_truths = tf.strided_slice(self.truths, [0, shift_height, shift_width, 0], \
                [conf.batch_size, shift_height + conf.valid_height, shift_width + conf.valid_width, conf.channel])
        
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(tf.square(mid_reconstruct - mid_truths))
            variable_summaries(self.loss)
            
        with tf.name_scope('original_loss'):
            self.original_loss = tf.reduce_mean(tf.square(mid_compres - mid_truths))
            variable_summaries(self.original_loss)

    
    def save(self, sess):
        # layer 1
        filename = os.path.join(self.conf.param_path, 'feature_extraction.npz')
        W = sess.run(self.W_1)
        b = sess.run(self.b_1)
        np.savez(filename, W=W, b=b)

        # layer 2
        filename = os.path.join(self.conf.param_path, 'feature_enhancement.npz')
        W = sess.run(self.W_2)
        b = sess.run(self.b_2)
        np.savez(filename, W=W, b=b)

        # layer 3
        filename = os.path.join(self.conf.param_path, 'mapping.npz')
        W = sess.run(self.W_3)
        b = sess.run(self.b_3)
        np.savez(filename, W=W, b=b)

        # layer 4
        filename = os.path.join(self.conf.param_path, 'reconstruction.npz')
        W = sess.run(self.W_4)
        b = sess.run(self.b_4)
        np.savez(filename, W=W, b=b)
