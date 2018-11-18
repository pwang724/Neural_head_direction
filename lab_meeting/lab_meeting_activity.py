import numpy as np
import pandas as pd
import utils
import os
import config
import pickle as pkl
import matplotlib.pyplot as plt
import analyze_receptive_field

def plot_activity(opts, save_name):
    save_path = opts.save_path
    data_name = opts.activity_name

    with open(os.path.join(save_path, data_name + '.pkl'), 'rb') as f:
        data_dict = pkl.load(f)

    states, predictions, labels = data_dict['states'], data_dict[
        'predictions'], data_dict['Y']

    row = 3
    tup = []
    for i in range(row):
        cur_state = np.array([s[i] for s in states])
        cur_pred = [p[i] for p in predictions]
        cur_label = labels[i, :, :]
        if i < 1:
            tup.append(('Prediction', cur_pred))
            tup.append(('Label', cur_label))
        else:
            tup.append(('', cur_pred))
            tup.append(('', cur_label))
    plot_name = save_name + '.png'
    plt.style.use('dark_background')
    utils.pretty_image(tup, col=2, row=row, save_name=plot_name, vmin=-.5,
                       vmax=.5)

if __name__ == '__main__':
    path = './lab_meeting/images/'

    d = './test/non_stationary/'
    opts = utils.load_parameters(d + '/parameters')
    opts.save_path = d
    plot_activity(opts, path + 'non_stationary_activity')

    d = './test/stationary/'
    opts = utils.load_parameters(d + '/parameters')
    opts.save_path = d
    plot_activity(opts, path + 'stationary_activity')

    d = './test/non_stationary/'
    opts = utils.load_parameters(d + '/parameters')
    points, activity, labels = analyze_receptive_field.get_activity(opts)
    analyze_receptive_field.plot_receptive_field(opts, points, activity,
                                                 plot_stationary=opts.stationary,
                         save_name=path + 'receptive_field_state')
    analyze_receptive_field.plot_receptive_field(opts, points, activity, plot_stationary=True,
                         save_name=path + 'receptive_field_support')

