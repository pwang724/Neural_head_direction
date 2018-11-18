class weight_names():
    def __init__(self):
        l = ['W_i_a', 'W_i_b', 'W_in', 'W_h', 'W_h_aa', 'W_h_ab', 'W_h_ba', 'W_h_bb', 'W_h_mask',
             'W_b', 'W_h_mask', 'W_out']
        for k in l:
            setattr(self, k, k)

class shared_config(object):
    """config for input parameters that are shared
    between both stationary and non-stationary inputs"""
    def __init__(self):
        self.rnn_size = 60
        self.state_size = 20
        self.bump_size = 6
        self.bump_std = 1.5

        self.n_input = 1000

        self.time_steps = 25
        self.weight_alpha = 1
        self.activity_alpha = .1

        self.noise = True
        self.noise_intensity = .3
        self.noise_density = .5

        self.file_name = 'model'
        self.weight_name = 'weight'
        self.activity_name = 'activity'
        self.parameter_name = 'parameters'
        self.image_folder = 'images'
        self.log_name = 'log'
        self.debug_weights = False

class stationary_input_config(shared_config):
    """config for stationary training"""
    def __init__(self):
        super(stationary_input_config, self).__init__()
        self.velocity = False

class stationary_model_config(stationary_input_config):
    def __init__(self):
        super(stationary_model_config, self).__init__()
        self.stationary = True

        self.epoch = 501
        self.batch_size = 20
        self.learning_rate = .001
        self.time_loss_start = 1
        self.time_loss_end = 25
        self.save_path = './test/stationary'
        self.load_checkpoint = False

class non_stationary_input_config(shared_config):
    """config for non-stationary training"""
    def __init__(self):
        super(non_stationary_input_config, self).__init__()
        self.velocity = True
        self.velocity_start = 5
        self.velocity_gap = 3
        self.velocity_use = 3
        self.velocity_max = 3
        self.velocity_size = 2
        self.time_steps = 25

class non_stationary_model_config(non_stationary_input_config):
    def __init__(self):
        super(non_stationary_model_config, self).__init__()

        self.stationary = False

        self.epoch = 401
        self.batch_size = 10
        self.learning_rate = .001
        self.time_loss_start = 1
        self.time_loss_end = 25

        self.load_weights = True #load pre-trained weights using stationary input
        self.fix_weights = False
        self.dir_weights = './test/stationary/weight.pkl'

        self.load_checkpoint = False #load checkpoint, overrides load_weights
        self.save_path = './test/non_stationary'

if __name__ == '__main__':
    a = non_stationary_model_config()
    print(a.__dict__)
    print(a.n_input)
    print(a.bump_size)