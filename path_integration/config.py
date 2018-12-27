import pickle as pkl
import os

class Options:

    def __init__(self):
        # input options
        self.max_steps_dir = 15
        self.max_angle_change = 45  # max 180
        self.d_step = 30
        self.d_angle = 20
        self.angle_format = 'onehot'  # trig, onehot, dist, cartesian
        self.output_format = 'cartesian'  # polar (radius, distance), cartesian (x-y, distance)
        self.agent_steps = 40
        self.n_examples = 10
        self.batch_size = 10
        self.bump_size = 6
        self.bump_std = 1.5
        self.label_units = 'rad'  # rad, deg
        self.use_gap = False
        self.step_gap = 3
        self.velocity = False
        self.direction = 'current'  # current, home
        self.zero_start = False
        self.stopping_prob = 0  # choose whether some steps will have no motion

        # test input options
        self.test = False
        self.test_type = '2-leg'  # full, 2-leg
        self.r0 = 20
        self.r1 = 10

        # network options
        self.network_state_size = 50
        self.rnn_type = 'tanh'  # sigmoid, tanh, relu, relu6 allowed
        self.continuous = False  # continuous rnn implementation
        self.positive_input = False
        self.network_loss = True
        self.n_epoch = int(2e2)
        self.separate_losses = False

        # i/o options
        self.name = "test"
        self.training_stage = 1
        self.folder = "test"
        self.weights_name = None

    def load(self):
        path = self.folder + "/" + self.get_path()
        with open(path, 'rb') as f:
            save_dict = pkl.load(f)
            for key, value in save_dict.items():
                self.__dict__[key] = value

    def save(self):
        path = self.folder + "/" + self.get_path()
        try:
            os.mkdir(self.folder)
        except FileExistsError:
            pass
        with open(path, 'wb+') as f:
            pkl.dump(self.__dict__, f)

    def get_path(self):
        return self.name + "_stage{}".format(self.training_stage)

