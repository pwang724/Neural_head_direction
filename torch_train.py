import torch
from numpy import random
import random
import time
import torch_model
import os
import pickle as pkl

import numpy as np
from datasets import inputs
import config
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
class InputDataset(Dataset):
    def __init__(self, opts):
        X, Y = inputs.create_inputs(opts)
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])

def torch2numpy(x):
    return x.detach().cpu().numpy()

def _initialize(save_path, reload, set_seed):
    if reload:  #use old model and new inputs
        opts = torch_model.load_config(save_path)
        new_input_config = config.inputConfig()
        opts.update(new_input_config)
    else:
        opts = config.inputConfig()
        opts.update(config.modelConfig())

    np.set_printoptions(precision=2)
    if set_seed:
        seed = opts.rng_seed
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)

    use_cuda = torch.cuda.is_available()
    if opts.ttype == 'float':
        ttype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
    else:
        ttype = torch.cuda.DoubleTensor if use_cuda else torch.DoubleTensor
    torch.set_default_tensor_type(ttype)

    dataset = InputDataset(opts)
    data_loader = DataLoader(dataset, batch_size= opts.batch_size, shuffle=True)
    net = torch_model.Simple_Model(config=opts, isize= dataset.X.shape[-1], osize = dataset.Y.shape[-1])

    if reload:
        net.load(name='net')
    print('[***Saving Variables***]')
    for name, param in net.named_parameters():
        if param.requires_grad:
            print('{0:20}: {1}'.format(name, param.data.shape))
    return opts, data_loader, net


def train(save_path, reload, set_seed = True):
    opts, data_loader, net = _initialize(save_path, reload, set_seed)
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0 * opts.learning_rate)

    n_epoch = opts.epoch
    logger = defaultdict(list)
    t_loss_start = opts.time_loss_start
    t_loss_end = opts.time_loss_end
    print("Starting training...")

    start_time = time.time()
    total_time = 0
    for ep in range(n_epoch):
        criterion = torch.nn.MSELoss()
        for x, y in data_loader:
            hidden = net.initialZeroState()
            optimizer.zero_grad()
            loss_activity = 0
            loss_weight = 0
            loss_pred = 0

            for t in range(x.shape[1]):
                xt = torch.Tensor(x[:,t,:])
                yt = torch.Tensor(y[:,t,:])
                hidden, out = net(xt, hidden)
                if t >= t_loss_start and t <= t_loss_end:
                    loss_activity += opts.activity_alpha * torch.mean(torch.pow(hidden,2))
                    loss_weight += opts.weight_alpha * torch.mean(torch.pow(net.i2h.weight,2)) #fix
                    loss_pred += criterion(out, yt)

            loss = loss_pred + loss_weight + loss_activity
            loss.backward()
            optimizer.step()

            logger['epoch'].append(ep)
            logger['loss'].append(torch2numpy(loss))
            logger['error_loss'].append(torch2numpy(loss_pred))
            logger['activity_loss'].append(torch2numpy(loss_activity))
            logger['weight_loss'].append(torch2numpy(loss_weight))

        pe = opts.print_epoch_interval
        se = opts.save_epoch_interval
        n_iter = opts.n_input // opts.batch_size
        if ep % pe == 0 and ep != 0:
            print('[' + '*' * 50 + ']')
            print('Epoch {:d}'.format(ep))
            print("Mean loss: {:0.2f}".format(np.mean(logger['loss'][-n_iter:])))
            print("Error loss: {0:.2f}, Weight loss: {1:.2f}, Activity loss: {2:.2f}".format(
                np.mean(logger['error_loss'][-n_iter:]),
                np.mean(logger['weight_loss'][-n_iter:]),
                np.mean(logger['activity_loss'][-n_iter:])))

            time_spent = time.time() - start_time
            total_time += time_spent
            start_time = time.time()
            print('Time taken {:0.1f}s'.format(total_time))
            print('Examples/second {:.1f}'.format(pe / time_spent))

        if ep % se == 0 and ep != 0:
            print("Saving files...")
            net.save('net', ep)
            net.save('net')
            with open(os.path.join(opts.save_path, 'loss_log.pkl'), 'wb') as f:
                pkl.dump(logger, f)

def evaluate(save_path, log):
    print("Starting testing...")

    opts, data_loader, net = _initialize(save_path, reload=True, set_seed =False)
    logger = defaultdict(list)


    for x, y in data_loader:
        hidden = net.initialZeroState()

        xs, ys, youts, hs = [], [], [], []
        for t in range(x.shape[1]):
            xt = torch.Tensor(x[:,t,:])
            yt = torch.Tensor(y[:,t,:])
            hidden, out = net(xt, hidden)
            xs.append(torch2numpy(xt))
            ys.append(torch2numpy(yt))
            youts.append(torch2numpy(out))
            hs.append(torch2numpy(hidden))

            logger['x'].append(np.array(xs))
            logger['y'].append(np.array(ys))
            logger['y_out'].append(np.array(youts))
            logger['h'].append(np.array(hs))
        break

    if log:
        #batch, time, neuron
        with open(os.path.join(opts.save_path, 'test_log.pkl'), 'wb') as f:
            pkl.dump(logger, f)
    return logger


if __name__ == "__main__":
    c = config.modelConfig()
    train(save_path=c.save_path, reload = False, set_seed=True)
    #
    # evaluate(save_path=c.save_path, log=True)