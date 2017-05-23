import tensorflow as tf
import numpy as np
import argparse
import math
import os
from models import *
from utils import *
from LIVE1 import *

def run(conf, data):
    print 'Testing Start.'
    for i in range(conf.epochs):
        sess = tf.Session()
        batch_truth, batch_compres, height, width = data.next_batch()
        conf.img_height = height
        conf.img_width = width
        truths = tf.placeholder(tf.float32, [conf.batch_size, height, width, conf.channel])
        compres = tf.placeholder(tf.float32, [conf.batch_size, height, width, conf.channel])
        conf.valid_height = int(height * 1.0)
        conf.valid_width = int(width * 1.0)
        model = ARCNN(conf, truths, compres)

        sess.run(tf.global_variables_initializer())
        
        data_dict = {compres:batch_compres, truths:batch_truth}

        reconstruct, cost, ori_cost = sess.run([model.F_4, model.loss, model.original_loss], feed_dict=data_dict)

        PSNR = 10.0 * math.log(1.0 / cost) / math.log(10.0)
        print 'Epoch: %d, Loss: %f, Original Loss: %f, PSNR: %f' % (i, cost, ori_cost, PSNR)
        
        save_img(batch_truth[0, :, :, :], os.path.join(conf.image_path, str(i + 1) + '_truth' + '.bmp'))
        save_img(batch_compres[0, :, :, :], os.path.join(conf.image_path, str(i + 1) + '_compress' + '.bmp'))
        save_img(reconstruct[0, :, :, :], os.path.join(conf.image_path, str(i + 1) + '_reconstruct' + '.bmp'))
    print 'Testing Completed.'



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--quality', type=int, default=10)
    parser.add_argument('--data_path', type=str, default='../data/ProcessedData/test')
    parser.add_argument('--param_path', type=str, default='../params')
    parser.add_argument('--image_path', type=str, default='../images')
    conf = parser.parse_args()
  
    data = LIVE1(conf.data_path, conf.quality)
    conf.epochs = 29
    conf.batch_size = 1
    conf.channel = 1
    conf.phase = 'test'

    conf = makepaths(conf) 
    
    run(conf, data)

