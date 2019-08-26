from functools import reduce 

class OMRModelHyperparameters:
    def __init__(
        self, 
        vocabulary_size):
        self.image_height = 128
        self.image_width = None
        self.minibatch_size = 16
        self.image_channels = 1
        self.convolutional_filter_number = [32, 64, 128, 256]
        self.convolutional_filter_size = [[3,3], [3,3], [3,3], [3,3]]
        self.convolutional_pooling_size = [[2,2], [2,2], [2,2], [2,2]]
        # self.width_reduction = reduce(lambda x, y: x*y, map(lambda x: x[1], self.conv_pooling_size))
        # self.height_reduction = reduce(lambda x, y: x*y, map(lambda x: x[0], self.conv_pooling_size))
        self.rnn_units_per_cell = 512
        self.rnn_cells = 2
        self.max_epochs = 10000
        self.keep_probability = .5
        self.vocabulary_size = vocabulary_size