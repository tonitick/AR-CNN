import tensorflow as tf
import numpy as np
import os
import tarfile
import sys
from six.moves import urllib

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 522000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 130500
HEIGHT = 32
WIDTH = 32
CHANNEL = 1

def read_img(filename_queue):

    class ImgRecord(object):
        pass
    result = ImgRecord()

    # Dimensions of the images in the CIFAR-10 dataset.
    # input format.
    result.height = HEIGHT
    result.width = WIDTH
    result.depth = CHANNEL
    image_bytes = result.height * result.width * result.depth
    # Every record consists of a label followed by the image, with a
    # fixed number of bytes for each.
    record_bytes_num = image_bytes * 2

    # Read a record, getting filenames from the filename_queue.
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes_num)
    result.key, value = reader.read(filename_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long.
    record_bytes = tf.decode_raw(value, tf.uint8)

    # The first bytes represent the label, which we convert from uint8->int32.
    depth_major_g = tf.reshape( \
            tf.strided_slice(record_bytes, [0], [image_bytes]), \
            [result.depth, result.width, result.height])

    # The remaining bytes after the label represent the image, which we reshape
    depth_major_c = tf.reshape( \
            tf.strided_slice(record_bytes, [image_bytes], \
                    [record_bytes_num]), \
            [result.depth, result.width, result.height])

    result.ground_truth = tf.transpose(depth_major_g, [2, 1, 0])

    result.compressed = tf.transpose(depth_major_c, [2, 1, 0])
  
    return result

def _generate_image_and_label_batch( \
            ground_truth, compressed, min_queue_examples, \
            batch_size, shuffle):

    num_preprocess_threads = 16
    if shuffle:
        truths, compres = tf.train.shuffle_batch(
                [ground_truth, compressed], \
                batch_size=batch_size, \
                num_threads=num_preprocess_threads, \
                capacity=min_queue_examples + 3 * batch_size, \
                min_after_dequeue=min_queue_examples)
    else:
        truths, compres = tf.train.batch(
                [ground_truth, compressed], \
                batch_size=batch_size, \
                num_threads=num_preprocess_threads, \
                capacity=min_queue_examples + 3 * batch_size)

    return truths, compres

class Data(object):
    def __init__(self, filenames, min_queue_examples, batch_size, shuffle):        
        for f in filenames:
            if not tf.gfile.Exists(f):
                raise ValueError('Failed to find file: ' + f)

        self.sess = tf.Session()

        filename_queue = tf.train.string_input_producer(filenames)
        self.read_input = read_img(filename_queue)

        self.read_input.ground_truth.set_shape([HEIGHT, WIDTH, CHANNEL])
        self.read_input.compressed.set_shape([HEIGHT, WIDTH, CHANNEL])
        self.truths, self.compres = _generate_image_and_label_batch(self.read_input.ground_truth, \
                self.read_input.compressed, min_queue_examples, batch_size, shuffle)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=self.sess)

    def next_batch(self):
        truths, compres = self.sess.run([self.truths, self.compres])
        truths = truths.astype(np.float32) / 255.0
        compres = compres.astype(np.float32) / 255.0
        return truths, compres


class BSDS500(object):
    def __init__(self, data_path, train_size, test_size, quality):
        data_dir = data_path
        
        filenames_train = [os.path.join(data_dir, 'train_' + str(quality) + '.bin'), \
                os.path.join(data_dir, 'test_' + str(quality) + '.bin')]
        min_fraction_of_examples_in_queue = 0.4
        min_queue_examples_train = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                min_fraction_of_examples_in_queue)
        self.train = Data(filenames_train, min_queue_examples_train, train_size, True)

        filenames_test = [os.path.join(data_dir, 'val_' + str(quality) + '.bin')]
        min_queue_examples_test = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        self.test = Data(filenames_test, min_queue_examples_test, test_size, False)
