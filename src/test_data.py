import tensorflow as tf
import numpy as np
import argparse
from BSDS500 import *
from PIL import Image

f = open("out.txt", "w")
data = BSDS500('../data/ProcessedData/train', 1, 10)
batch_y, batch_x = data.test.next_batch(1)
y = batch_y.reshape([32 * 32])
x = batch_x.reshape([32 * 32])
y = y.astype(np.int32)
x = x.astype(np.int32)
for i in range(32 * 32):
    print >> f, y[i]
print >>f
for i in range(32*32):
    print >>f, x[i]
