class shared_config(object):
    """config for input parameters that are shared
    between both stationary and non-stationary inputs"""
    def __init__(self):
        self.state_size = 20
        self.bump_size = 6
        self.bump_std = 1.5

        self.noise = True
        self.noise_intensity = .3
        self.noise_density = .5

        self.file_name = 'model'
        self.weight_name = 'weight'
        self.activity_name = 'activity'
        self.parameter_name = 'parameters'
        self.image_folder = 'images'
        self.log_name = 'log'

class stationary_input_config(shared_config):
    """config for stationary training"""
    def __init__(self):
        super(stationary_input_config, self).__init__()
        self.n_input = 1000
        self.time_steps = 25
        self.velocity = False

class stationary_model_config(stationary_input_config):
    def __init__(self):
        super(stationary_model_config, self).__init__()
        self.stationary = True
        self.rnn_size = 70
        self.epoch = 101
        self.batch_size = 25
        self.test_batch_size = 5
        self.learning_rate = .001
        self.time_loss_start = 5
        self.time_loss_end = 20
        self.save_path = './test/stationary'
        self.load_checkpoint = False
        self.weight_alpha = 1
        self.activity_alpha = .1

class non_stationary_input_config(shared_config):
    """config for non-stationary training"""
    def __init__(self):
        super(non_stationary_input_config, self).__init__()
        self.n_input = 1000
        self.velocity = True
        self.velocity_start = 5
        self.velocity_gap = 3
        self.velocity_max = 1
        self.velocity_size = self.velocity_max * 2
        self.velocity_use = [1]
        self.time_steps = 50

class non_stationary_model_config(non_stationary_input_config):
    def __init__(self):
        super(non_stationary_model_config, self).__init__()

        self.stationary = False
        self.rnn_size = 60

        self.epoch = 201
        self.batch_size = 20
        self.test_batch_size = 5
        self.learning_rate = .001
        self.time_loss_start = 1
        self.time_loss_end = 25
        self.weight_alpha = 1
        self.activity_alpha = .1

        self.load_weights = True #load pre-trained weights using stationary input
        self.fix_weights = True
        self.dir_weights = './gold/stationary/weight.pkl'

        self.load_checkpoint = False #load checkpoint, overrides load_weights
        self.save_path = './test/non_stationary'

if __name__ == '__main__':
    a = non_stationary_model_config()
    print(a.__dict__)
    print(a.n_input)
    print(a.bump_size)