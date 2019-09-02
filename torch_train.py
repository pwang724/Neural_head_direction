import torch
from numpy import random
from torch import optim
import random
import time
import torch_model
import matplotlib.pyplot as plt
import os
import pickle as pkl

import numpy as np
import inputs
import config
from collections import defaultdict

from torch.utils.data import Dataset, DataLoader
class InputDataset(Dataset):
    def __init__(self, opts):
        X, Y, N, _ = inputs.create_inputs(opts)
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


def torch2numpy(x):
    return x.detach().cpu().numpy()

def train(save_path, reload, set_seed = True):
    if reload:  #use old model and new inputs
        opts = torch_model.Simple_Model.load_config(save_path)
        new_input_config = config.shared_config()
        opts.update(new_input_config)
    else:
        opts = config.shared_config()
        opts.update(config.non_stationary_model_config())
        opts.update(config.non_stationary_input_config())

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

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    dataset = InputDataset(opts)
    train_loader = DataLoader(dataset, batch_size= opts.batch_size, shuffle=True)
    net = torch_model.Simple_Model(config=opts, isize= dataset.X.shape[-1], osize = dataset.Y.shape[-1])
    optimizer = torch.optim.Adam(net.parameters(), lr=1.0 * opts.learning_rate)

    if reload:
        net.load(name='net')
    print('[***Saving Variables***]')
    for name, param in net.named_parameters():
        if param.requires_grad:
            print('{0:20}: {1}'.format(name, param.data.shape))

    n_epoch = opts.epoch
    logger = defaultdict(list)
    t_loss_start = opts.time_loss_start
    t_loss_end = opts.time_loss_end
    print("Starting training...")

    start_time = time.time()
    total_time = 0
    for ep in range(n_epoch):
        criterion = torch.nn.MSELoss()
        for x, y in train_loader:
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









def evaluate(save_path, BS):
    # net, c = initialize(save_path=save_path, reload=True, batch_size= BS, set_seed= False)
    #
    envs = []
    for _ in range(BS):
        envs.append(input.Environment(c))

    reward = np.zeros(BS)
    rewards, values, logprobs, action_probs, actions, mods, tderrs = [], [], [], [], [], [], []

    net.initializeState()
    inputs = [env.reset()[0] for env in envs]
    episode_length = np.min([env.length for env in envs])
    print("Starting testing...")
    for numstep in range(episode_length):
        inputs = np.stack(inputs)
        inputsC = torch.Tensor(inputs)

        action_prob, v = net(inputsC)
        distrib = torch.distributions.Categorical(action_prob)
        action = distrib.sample()
        logprobs.append(distrib.log_prob(action))
        numactionschosen = action.data.cpu().numpy()

        for nb, env in enumerate(envs):
            inputs[nb], reward[nb], done = env.step(numactionschosen[nb])

        mods.append(net.mod)
        rewards.append(reward.copy())
        values.append(v)
        actions.append(action)
        action_probs.append(action_prob)
    print("Done testing...")

    R = torch.zeros(BS, requires_grad=False)
    for numstepb in reversed(range(episode_length)):
        R = c.DISCOUNT * R + torch.Tensor(rewards[numstepb])  # Tensor requires_grad default is False
        tdErr = R - values[numstepb].view(BS)
        tderrs.append(tdErr)

    data_transform_F = lambda x: np.stack([a.squeeze().cpu().detach().numpy() for a in x])

    out = {}
    out['R'] = np.stack(rewards) # TSTEP X BATCH
    out['V'] = data_transform_F(values)
    out['A'] = data_transform_F(actions) # TSTEP X BATCH
    out['A_P'] = data_transform_F(action_probs) # TSTEP X BATCH X ACTION_STATE_SIZE
    out['M'] = data_transform_F(mods)  # TSTEP X BATCH
    out['V_ERR'] = data_transform_F(tderrs)
    return envs, out


if __name__ == "__main__":
    c = config.non_stationary_model_config()
    train(save_path=c.save_path, reload = False, set_seed=True)