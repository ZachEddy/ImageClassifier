import numpy as np
from src.volume import volume
# a class to manage the ReLu activation layer
class relu_layer: 
	def __init__(self, in_height, in_width, in_depth):
		# define input dimensions
		self.in_height = in_height
		self.in_width = in_width
		self.in_depth = in_depth
		
		# define output dimensions (they're the same, but needed )
		self.out_height = in_height
		self.out_width = in_width
		self.out_depth = in_depth

		# vectorize the ReLU activation function so it can easily be mapped over neuron volumes
		self.relu_volume = np.vectorize(self.relu_single)

	# run the activation function on a single neuron
	def relu_single(self, neuron):
		# the activation function is simply max(0,x). consider making it shakey at some point down the road
		return max(0, neuron)

	# the feed-forward function for a ReLu layer
	def forward(self, input_volume):
		return volume(self.relu_volume(input_volume.volume_slices))