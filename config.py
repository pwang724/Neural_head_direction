class weight_names():
    def __init__(self):
        # simple as possible to start
        l = ['W_in', 'W_h', 'W_h_mask', 'W_h_bias',
             'W_out', 'W_out_bias', 'ei_mask']
        for k in l:
            setattr(self, k, k)

class BaseConfig(object):
    def __init__(self):
        self.rng_seed = 0
        self.model_name = 'model'
        self.weight_name = 'weight'
        self.activity_name = 'activity'
        self.parameter_name = 'parameters'
        self.image_folder = 'images'
        self.log_name = 'log'

    def update(self, new_config):
        self.__dict__.update(new_config.__dict__)

    def __str__(self):
        return str(self.__dict__)

class inputConfig(BaseConfig):
    """config for non-stationary training"""
    def __init__(self):
        super(inputConfig, self).__init__()
        self.state_size = 36  # size of position input, not the hidden layer
        self.n_input = 1000
        self.time_steps = 50

        self.input_mode = 'trig'  # ['bump', 'trig', 'onehot']
        self.output_mode = 'trig'  # ['bump', 'trig', 'onehot']
        self.discrete = True # discrete positions and velocities

        self.non_negative_input = True
        self.non_negative_output = True

        self.bump_size = 5
        self.bump_std = 1

        self.noise = True
        self.noise_intensity = .05
        self.noise_density = .5

        self.velocity = True
        self.velocity_max = 30 #degrees
        self.velocity_start = 5
        self.velocity_gap = 3
        self.stop_probability = 0
        self.velocity_onehot = True

        #DEPRECATED
        # self.linear_track = True
        # self.subtrack = False
        # self.subtrack_minlen = 18
        # self.subtrack_maxlen = 72
        # self.n_env = 10
        # self.rescale_env = True  # make the boundaries 0 and 360 deg for all environments
        # self.nav_output = False
        # self.home = None
        # self.boundary_velocity = False
        # self.grid_input = False


class modelConfig(BaseConfig):
    def __init__(self):
        super(modelConfig, self).__init__()
        self.learning_rate = .001
        self.batch_size = 20
        self.test_batch_size = 1000
        self.epoch = 201
        self.time_loss_start = 5
        self.time_loss_end = 20

        self.rnn_size = 100
        self.dropout = False
        self.dropout_rate = .2
        self.weight_alpha = .1
        self.activity_alpha = .1
        self.activation_fn = 'relu'  # [relu, relu, tanh]
        self.constrained = False

        self.EI = False
        self.prop_ex = .8
        # self.EI_in = False
        # self.EI_h = False
        # self.EI_out = False

        self.reload = False  # load checkpoint, overrides load_weights
        self.save_path = './_DATA/test'
        self.use_velocity = True

        self.ttype = 'float'
        self.print_epoch_interval = 5
        self.save_epoch_interval = 100

        self.debug_weights = False

        #DEPRECATED
        # self.load_weights = False  # load pre-trained weights using stationary input
        # self.fix_weights = False
        # self.losses = 'error'  # choose from error, activity, weights, full

if __name__ == '__main__':
    a = inputConfig()
    print(a.__dict__)
    print(a.n_input)
    print(a.bump_size)
