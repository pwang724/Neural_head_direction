class weight_names():
    def __init__(self):
        # simple as possible to start
        l = ['W_in', 'W_h', 'W_h_mask', 'W_h_bias',
             'W_out', 'W_out_bias', 'ei_mask']
        for k in l:
            setattr(self, k, k)

class shared_config(object):
    """config for input parameters that are shared
    between both stationary and non-stationary inputs"""
    def __init__(self):
        self.state_size = 36  # size of position input, not the hidden layer
        self.rnn_size = 100
        self.bump_size = 1
        self.bump_std = 1.5

        self.n_input = 1000
        self.time_steps = 25

        self.noise = True
        self.noise_intensity = .1
        self.noise_density = .5

        self.mask = True

        self.model_name = 'model'
        self.weight_name = 'weight'
        self.activity_name = 'activity'
        self.parameter_name = 'parameters'
        self.image_folder = 'images'
        self.log_name = 'log'
        self.debug_weights = False
        self.testing = False

        self.activation_fn = 'relu'  # [relu, relu6, tanh]
        self.output_mode = 'trig'  # ['bump', 'trig', 'scalar']
        self.linear_track = True
        self.input_mode = 'bump'  # ['bump', 'scalar', 'trig']
        self.bump_in_network = False

        self.dropout = False
        self.dropout_rate = .2

        self.EI = False
        self.prop_ex = .8

        self.nav_output = False
        self.home = None

        self.nonneg_input = False
        self.EI_in = False
        self.EI_h = False
        self.EI_out = False

class stationary_input_config(shared_config):
    """config for stationary training"""
    def __init__(self):
        super(stationary_input_config, self).__init__()
        self.velocity = False

class stationary_model_config(stationary_input_config):
    def __init__(self):
        super(stationary_model_config, self).__init__()
        self.stationary = True
        self.weight_alpha = .05
        self.activity_alpha = .01

        self.epoch = 300
        self.batch_size = 20
        self.learning_rate = .001
        self.time_loss_start = 4
        self.time_loss_end = 25
        self.save_path = './training/stationary'
        self.load_checkpoint = False
        self.losses = 'error'  # choose from error, activity, weights, full

class non_stationary_input_config(shared_config):
    """config for non-stationary training"""
    def __init__(self):
        super(non_stationary_input_config, self).__init__()
        self.velocity = True
        self.velocity_start = 5
        self.velocity_gap = 3
        self.velocity_step = 1
        self.velocity_min = 1
        self.velocity_max = 2
        self.velocity_size = 2
        self.time_steps = 25

        self.boundary_velocity = False
        self.correlated_path = True
        self.grid_input = False

        # subtrack, for use with ipt size 72 and up
        self.subtrack = False
        self.subtrack_minlen = 18
        self.subtrack_maxlen = 72
        self.n_env = 10
        self.rescale_env = True  # make the boundaries 0 and 360 deg for all environments

class non_stationary_model_config(non_stationary_input_config):
    def __init__(self):
        super(non_stationary_model_config, self).__init__()

        self.stationary = False
        self.weight_alpha = .5
        self.activity_alpha = .1

        self.epoch = 401
        self.batch_size = 20
        self.learning_rate = .001
        self.time_loss_start = 5
        self.time_loss_end = 20
        self.losses = 'error'  # choose from error, activity, weights, full

        # These don't seem necessary
        # self.load_weights = False  # load pre-trained weights using stationary input
        # self.fix_weights = False

        self.load_checkpoint = False  # load checkpoint, overrides load_weights
        self.save_path = './test/non_stationary'
        self.use_velocity = True

if __name__ == '__main__':
    a = non_stationary_model_config()
    print(a.__dict__)
    print(a.n_input)
    print(a.bump_size)
