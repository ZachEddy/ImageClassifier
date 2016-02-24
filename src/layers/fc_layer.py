import numpy as np
from src.volume import volume
# a class to manage the ReLu activation layer
class fc_layer: 
	def __init__(self, in_height, in_width, in_depth, neuron_count):
		# define input dimensions
		self.in_height = in_height
		self.in_width = in_width
		self.in_depth = in_depth

		# define output dimensions
		self.out_height = 1
		self.out_width = neuron_count
		self.out_depth = 1

		# setup the weights
		self.initalize_weights()

	# a function to initialize the weights of the fully connected layer
	def initalize_weights(self):
		weights = []
		weight_count = self.in_height * self.in_width * self.in_depth
		# initialize weights for each neuron in the network
		for i in range(self.out_width):
			weight_gen = np.random.randn(weight_count) * np.sqrt(2.0 / weight_count)
			weight_gen = np.reshape(weight_gen, (self.in_depth, self.in_height, self.in_width))
			weights.append(weight_gen)
		# set as an instance variable for later methods
		self.weights = weights

	# a function that feeds an input volume through the network
	def forward(self, input_volume):
		output = np.array([])
		# each matrix of weights corresponds to a slice in the output volume
		for weight in self.weights:
			output = np.append(output, np.sum(weight * input_volume.volume_slices) + 0.1)
		
		# maybe redefine which axix the wights are on (height, width, or depth)?
		return volume(np.reshape(output, (1,1, self.out_width)))