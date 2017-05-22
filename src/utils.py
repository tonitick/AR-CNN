import tensorflow as tf
import numpy as np
from PIL import Image
import os

def variable_summaries(var):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def makepaths(conf):
    conf.ckpt_path = os.path.join(conf.ckpt_path, "bs=%d_quality=%d"%(conf.batch_size, conf.quality))
    if not os.path.exists(conf.ckpt_path):
        os.makedirs(conf.ckpt_path)

    conf.param_path_path = os.path.join(conf.param_path_path, "bs=%d_quality=%d"%(conf.batch_size, conf.quality))
    if not os.path.exists(conf.param_path_path):
        os.makedirs(conf.param_path_path)

    conf.summary_path = os.path.join(conf.summary_path, "bs=%d_quality=%d"%(conf.batch_size, conf.quality))
    if tf.gfile.Exists(conf.summary_path):
        tf.gfile.DeleteRecursively(conf.summary_path)
    tf.gfile.MakeDirs(conf.summary_path)

    return conf

def save_img(arr, path):
    print arr.shape
    arr = arr * 255
    arr = arr.reshape(arr.shape[0], arr.shape[1])
    img = Image.fromarray(arr.astype(np.uint8))
    img.save(path)