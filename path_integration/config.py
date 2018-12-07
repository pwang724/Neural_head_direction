class Options:

    def __init__(self):
        self.max_steps_dir = 5
        self.max_angle_change = 45
        self.state_size = 20
        self.angle_format = 'trig'  # trig, onehot, dist
        self.timesteps = 20
        self.n_batch = 10
        self.n_examples = 10
        self.bump_size = 6
        self.bump_std = 1.5

        self.name = ""
        self.weights_name = None
        self.folder = "test"

