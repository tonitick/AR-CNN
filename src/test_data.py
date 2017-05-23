import numpy as np
import tensorflow as tf
import argparse
from BSDS500 import *
from LIVE1 import *
from utils import *
from PIL import Image

data = BSDS500('../data/ProcessedData/train', 10)
for i in range(1):
    batch_y, batch_x = data.train.next_batch(1305)
    print batch_y

truth = np.zeros([312, 472])
compres = np.zeros([312, 472])

truth_integrated = np.zeros([312, 472], np.float32)
compres_integrated = np.zeros([312, 472], np.float32)
for i in range(29):
    for j in range(45):
        truth_integrated[(i * 10):(i * 10 + 32), (j * 10):(j * 10 + 32)] = batch_y[(i * 45 + j), :, :, 0]
        compres_integrated[(i * 10):(i * 10 + 32), (j * 10):(j * 10 + 32)] = batch_x[(i * 45 + j), :, :, 0]

save_img(truth_integrated, './train_truth.bmp')
save_img(compres_integrated, './train_compres.bmp')

data = LIVE1('../data/ProcessedData/test', 10)
truth, compres, height, width = data.next_batch()

save_img(truth[0, :, :, :], './test_truth.bmp')
save_img(compres[0, :, :, :], './test_compres.bmp')
