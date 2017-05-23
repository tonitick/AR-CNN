import numpy as np
import os

CHANNEL = 1

import numpy as np
import os

HEIGHT = 32
WIDTH = 32
CHANNEL = 1
NUM_TRAIN = 261000
NUM_TEST = 130500


class Train(object):
    def __init__(self, data_dir, quality):
        self.quality = quality
        self.data_dir = data_dir
        self.filenames = ['train', 'test']
        self.current_index = 0
        self.currnet_file = 1

    def swap_file(self):
        if self.currnet_file == 0:
            self.currnet_file = 1
        else:
            self.currnet_file = 0

        data_file = os.path.join(self.data_dir, self.filenames[self.currnet_file] + '_' + str(self.quality) + '.bin')
        self.data = np.fromfile(data_file, dtype=np.uint8)


    def next_batch(self, batch_size):
        pixels = HEIGHT * WIDTH * CHANNEL

        truth_batch = np.zeros([batch_size, HEIGHT, WIDTH, CHANNEL])
        compres_batch = np.zeros([batch_size, HEIGHT, WIDTH, CHANNEL])
        for i in range(batch_size):
            if self.current_index == 0:
                self.swap_file()
            
            truth_depth_major = self.data[(self.current_index * 2 * pixels):((self.current_index * 2 + 1) * pixels)].reshape([CHANNEL, WIDTH, HEIGHT])
            truth = truth_depth_major.transpose((2, 1, 0))
            compres_depth_major = self.data[((self.current_index * 2 + 1) * pixels):((self.current_index * 2 + 2) * pixels)].reshape([CHANNEL, WIDTH, HEIGHT])
            compres = compres_depth_major.transpose((2, 1, 0))

            truth_batch[i, :, :, :] = truth.astype(np.float32) / 255.0
            compres_batch[i, :, :, :] = compres.astype(np.float32) / 255.0

            self.current_index = (self.current_index + 1) % NUM_TRAIN

        return truth_batch, compres_batch


class Test(object):
    def __init__(self, data_dir, quality):
        self.quality = quality
        self.data_dir = data_dir
        self.filename = 'val'
        self.current_index = 0
        data_file = os.path.join(data_dir, self.filename + '_' + str(quality) + '.bin')
        self.data = np.fromfile(data_file, dtype=np.uint8)


    def next_batch(self, batch_size):
        pixels = HEIGHT * WIDTH * CHANNEL

        truth_batch = np.zeros([batch_size, HEIGHT, WIDTH, CHANNEL], np.float32)
        compres_batch = np.zeros([batch_size, HEIGHT, WIDTH, CHANNEL], np.float32)
        for i in range(batch_size):
            truth_depth_major = self.data[(self.current_index * 2 * pixels):((self.current_index * 2 + 1) * pixels)].reshape([CHANNEL, WIDTH, HEIGHT])
            truth = truth_depth_major.transpose((2, 1, 0))
            compres_depth_major = self.data[((self.current_index * 2 + 1) * pixels):((self.current_index * 2 + 2) * pixels)].reshape([CHANNEL, WIDTH, HEIGHT])
            compres = compres_depth_major.transpose((2, 1, 0))

            truth_batch[i, :, :, :] = truth.astype(np.float32) / 255.0
            compres_batch[i, :, :, :] = compres.astype(np.float32) / 255.0

            self.current_index = (self.current_index + 1) % NUM_TEST

        return truth_batch, compres_batch


class BSDS500(object):
    def __init__(self, data_path, quality):
        self.train = Train(data_path, quality)
        self.test = Test(data_path, quality)