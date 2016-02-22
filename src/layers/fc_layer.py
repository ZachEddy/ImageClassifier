import numpy as np
from src.volume import volume
# a class to manage the ReLu activation layer
class relu_layer: 
	def __init__(self, in_height, in_width, in_depth, neuron_count):
		# define input dimensions
		self.in_height = in_height
		self.in_width = in_width
		self.in_depth = in_depth

		# define output dimensions
		self.out_height = neuron_count
		self.out_width = 1
		self.out_depth = 1

		def initalize_weights(self):
			weights = []
			weight_count = self.in_height * self.in_width * self.in_depth
			for i in self.neuron_count:
				weight_gen = np.random.randn(weight_count) * np.sqrt(2.0 / weight_count)
				weights.append(weight_gen)

		def forward(self, input_volume):
			# I don't do anything yet
			return
