import os
import utils
import numpy as np
import pickle
import utils

root_dir = '/Users/peterwang/Desktop/PYTHON/Neural_head_direction/experiments/vary_randomize_ab_ba'
image_dir = os.path.join(root_dir, 'images')
subfile = 'moving'
xe_loss, loss, config = utils.load_results(root_dir, subfile)

if not os.path.exists(image_dir):
    os.mkdir(image_dir)

xe_loss = np.array(xe_loss).transpose()
loss = np.array(loss).transpose()
tup = [('xe loss', xe_loss), ('total loss', loss)]
legends = ['no weight loss', 'weight loss']
utils.subplot_easy(tup, legends, col=2, row=2, save_name= os.path.join(image_dir,
                                                                       'summary_' + subfile))