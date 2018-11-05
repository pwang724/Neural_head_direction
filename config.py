class shared_input_config(object):
    """config for input parameters that are shared
    between both stationary and non-stationary inputs"""
    def __init__(self):
        self.state_size = 20
        self.bump_size = 6
        self.bump_std = 1.5

        self.noise = True
        self.noise_intensity = .3
        self.noise_density = .5

class stationary_input_config(shared_input_config):
    """config for stationary training"""
    def __init__(self):
        super(stationary_input_config, self).__init__()
        self.n_input = 1000
        self.time_steps = 20
        self.velocity = False

class stationary_model_config(stationary_input_config):
    def __init__(self):
        super(stationary_model_config, self).__init__()
        self.stationary = True
        self.rnn_size = 60
        self.epoch = 200
        self.batch_size = 5
        self.test_batch_size = 5
        self.learning_rate = .001
        self.time_loss_start = 5
        self.time_loss_end = 20
        self.save_path = './test/stationary'
        self.file_name = 'tf'
        self.load_checkpoint = False

class non_stationary_input_config(shared_input_config):
    """config for non-stationary training"""
    def __init__(self):
        super(non_stationary_input_config, self).__init__()
        self.n_input = 1000
        self.velocity = True
        self.velocity_start = 5
        self.velocity_gap = 3
        self.velocity_max = 3
        self.velocity_size = self.velocity_max * 2
        self.velocity_use = [1, 3]
        self.time_steps = 50

class non_stationary_model_config(non_stationary_input_config):
    def __init__(self):
        super(non_stationary_model_config, self).__init__()

        self.stationary = False
        self.rnn_size = 60

        self.epoch = 100
        self.batch_size = 5
        self.test_batch_size = 5
        self.learning_rate = .001
        self.time_loss_start = 1
        self.time_loss_end = 50

        self.load_weights = True #load pre-trained weights using stationary input
        self.fix_weights = True
        self.dir_weights = './test/stationary/tf.pkl'

        self.load_checkpoint = True #load checkpoint, overrides load_weights
        self.save_path = './test/non_stationary'
        self.file_name = 'tf'

if __name__ == '__main__':
    a = non_stationary_model_config()
    print(a.__dict__)
    print(a.n_input)
    print(a.bump_size)