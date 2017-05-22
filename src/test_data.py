import numpy as np
import argparse
from LIVE1 import *
from utils import *

data = LIVE1('../data/ProcessedData/test', 10)
truth, compres = data.next_batch()

save_img(truth[0, :, :, :], './truth.bmp')
save_img(compres[0, :, :, :], './compres.bmp')





