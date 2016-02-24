import numpy as np
from src.volume import volume
# a class to manage the tanh activation layer
class tanh_layer: 
	def __init__(self, in_height, in_width, in_depth):
		# define input dimensions
		self.in_height = in_height
		self.in_width = in_width
		self.in_depth = in_depth
		
		# define output dimensions (they're the same, but needed for the next layer)
		self.out_height = in_height
		self.out_width = in_width
		self.out_depth = in_depth

		# vectorize the tanh activation function so it can easily be mapped over neuron volumes
		self.tanh_volume = np.vectorize(self.tanh_single)

	# a function that values the tanh activation function on a single neuron in the volume
	def tanh_single(self, neuron):
		# the activation function is tanh(x) of the neuron
		return np.tanh(neuron)

	# a function that feeds the input volume through the network
	def forward(self, input_volume):
		return volume(self.tanh_volume(input_volume.volume_slices))