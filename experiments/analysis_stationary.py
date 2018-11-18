import experiments.experiment_stationary as exp
import os
import numpy as np
import pickle
import utils

def _load_results(root_dir):
    dir = os.path.join(root_dir, 'files')
    dirs = [os.path.join(root_dir, dir, n) for n in os.listdir(dir)]
    dirs = sorted(dirs)
    xe_loss = []
    loss = []
    config = []
    for i, d in enumerate(dirs):
        log_name = os.path.join(d, 'log.pkl')
        with open(log_name, 'rb') as f:
            log = pickle.load(f)
        config.append(utils.load_parameters(os.path.join(d, 'parameters')))

        xe_loss.append(log['xe_loss'])
        loss.append(log['loss'])
    return xe_loss, loss, config

root_dir = '/Users/peterwang/Desktop/PYTHON/Neural_head_direction/experiments/vary_batch'
image_dir = os.path.join(root_dir, 'images')
xe_loss, loss, config = _load_results(root_dir)

if not os.path.exists(image_dir):
    os.mkdir(image_dir)

xe_loss = np.array(xe_loss).transpose()
loss = np.array(loss).transpose()
tup = [('xe loss', xe_loss), ('total loss', loss)]
legends = [c.batch_size for c in config]
utils.subplot_easy(tup, legends, col=2, row=2, save_name= os.path.join(image_dir, 'summary'))