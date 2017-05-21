import tensorflow as tf
import numpy as np
import argparse
import math
from models import *
from utils import *
from BSDS500 import *

def run(conf, data):
    sess = tf.Session()

    print 'Model Defining...'
    model = ARCNN(conf)

    trainer = tf.train.RMSPropOptimizer(1e-3)
    gradients = trainer.compute_gradients(model.loss)
    clipped_gradients = [(tf.clip_by_value(_[0], -conf.grad_clip, conf.grad_clip), _[1]) for _ in gradients]
    optimizer = trainer.apply_gradients(clipped_gradients)

    saver = tf.train.Saver()

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(conf.summary_path, sess.graph)
    
    if os.path.exists(conf.ckpt_path):
        ckpt = tf.train.get_checkpoint_state(conf.ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:  
            saver.restore(sess, ckpt.model_checkpoint_path)
            print 'Model Restored'
        else:
            sess.run(tf.global_variables_initializer())
            print 'Initialize Variables'
    else:
        sess.run(tf.global_variables_initializer())
        print 'Initialize Variables'

    print 'Training...'
    for i in range(conf.epochs):
        for j in range(conf.num_batches):
            batch_truth, batch_compres = data.train.next_batch(conf.batch_size)

            data_dict = {model.compres:batch_compres, model.truths:batch_truth}

            _, cost, summary = sess.run([optimizer, model.loss, merged], feed_dict=data_dict)

        writer.add_summary(summary, i)

        PSNR = 10.0 * math.log(cost) / math.log(10.0)
        print 'Epoch: %d, Cost: %f, PSNR: %f' % (i, cost, PSNR)
    
    saver.save(sess, conf.ckpt_path + '/model.ckpt')


    print 'Validating...'
    num_val_epochs = conf.num_val / conf.test_size + 1
    for i in range(num_val_epochs):

        batch_truth, batch_compres = data.test.next_batch(conf.test_size)

        data_dict = {model.compres:batch_compres, model.truths:batch_truth}

        cost = sess.run(model.loss, feed_dict=data_dict)

        PSNR = 10.0 * math.log(cost) / math.log(10.0)
        print 'Epoch: %d, Cost: %f, PSNR: %f' % (i, cost, PSNR)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--quality', type=int, default=10)
    parser.add_argument('--grad_clip', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='../data/ProcessedData/train')
    parser.add_argument('--summary_path', type=str, default='../logs')
    parser.add_argument('--ckpt_path', type=str, default='../ckpts')
    conf = parser.parse_args()
  
    data = BSDS500(conf.data_path, conf.batch_size, conf. test_size, conf.quality)
    conf.num_classes = 10
    conf.img_height = 32
    conf.img_width = 32
    conf.channel = 1
    conf.valid_height = 20
    conf.valid_width = 20
    conf.num_train = 522000
    conf.num_val = 130500
    conf.num_batches = 10
    # conf.num_batches = num_train / conf.batch_size + 1

    conf = makepaths(conf) 
    
    run(conf, data)

