import numpy as np
import tensorflow as tf
import argparse
from BSDS500 import *
from LIVE1 import *
from utils import *
from PIL import Image

f = open("out.txt", "w")
data = BSDS500('../data/ProcessedData/train', 1, 1305, 10)
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
batch_y, batch_x = data.test.next_batch()
truth = np.zeros([312, 472])
compres = np.zeros([312, 472])

truth_integrated = np.zeros([312, 472])
compres_integrated = np.zeros([312, 472])
for i in range(0, 29):
    for j in range(0, 45):
        truth_integrated[(i * 10):(i * 10 + 32), (j * 10):(j * 10 + 32)] = batch_y[(i * 45 + j), :, :, 0]
        compres_integrated[(i * 10):(i * 10 + 32), (j * 10):(j * 10 + 32)] = batch_x[(i * 45 + j), :, :, 0]

save_img(truth_integrated, './train_truth.bmp')
save_img(compres_integrated, './train_compres.bmp')

data = LIVE1('../data/ProcessedData/test', 10)
truth, compres = data.next_batch()

save_img(truth[0, :, :, :], './test_truth.bmp')
save_img(compres[0, :, :, :], './test_compres.bmp')





