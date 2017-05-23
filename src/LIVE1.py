import numpy as np
import os

CHANNEL = 1

class LIVE1(object):
    def __init__(self, data_path, quality):       
        self.names = ['bikes', 'building2', 'buildings', 'caps', 'carnivaldolls', \
                'cemetry', 'churchandcapitol', 'coinsinfountain', 'dancers', 'flowersonih35', \
                'house', 'lighthouse', 'lighthouse2', 'manfishing', 'monarch', 'ocean', \
                'paintedhouse', 'parrots', 'plane', 'rapids', 'sailing1', 'sailing2', \
                'sailing3', 'sailing4', 'statue', 'stream', 'studentsculpture', 'woman', 
                'womanhat']

        self.data_dir = data_path

        self.quality = quality

        self.current_index = 0

        self.num = len(self.names)

    def next_batch(self):
        data_file = os.path.join(self.data_dir, self.names[self.current_index] + '_' + str(self.quality) + '.bin')
        data = np.fromfile(data_file, dtype=np.uint8)
        self.current_index = (self.current_index + 1) % self.num
        
        height_char = data[0:4].astype(np.uint32)
        width_char = data[4:8].astype(np.uint32)
        
        # conver to int32
        height = int(height_char[0] + height_char[1] * 256)
        width = int(width_char[0] + width_char[1] * 256)   
        
        truth_depth_major = data[8:(8 + height * width)].reshape([1, CHANNEL, width, height])
        compres_depth_major = data[(8 + height * width):(8 + height * width * 2)].reshape([1, CHANNEL, width, height])

        truth = truth_depth_major.transpose((0, 3, 2, 1))
        compres = compres_depth_major.transpose((0, 3, 2, 1))

        truth = truth.astype(np.float32) / 255.0
        compres = compres.astype(np.float32) / 255.0

        return truth, compres, height, width