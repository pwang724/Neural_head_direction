import experiments.experiment_stationary as exp
import os
import numpy as np
import pickle
import utils

root_dir = '/Users/peterwang/Desktop/PYTHON/Neural_head_direction/experiments/vary_batch'
image_dir = os.path.join(root_dir, 'images')
xe_loss, loss, config = utils.load_results(root_dir)

if not os.path.exists(image_dir):
    os.mkdir(image_dir)

xe_loss = np.array(xe_loss).transpose()
loss = np.array(loss).transpose()
tup = [('xe loss', xe_loss), ('total loss', loss)]
legends = [c.batch_size for c in config]
utils.subplot_easy(tup, legends, col=2, row=2, save_name= os.path.join(image_dir, 'summary'))