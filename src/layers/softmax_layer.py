import numpy as np
from src.volume import volume
# a class to manage the tanh activation layer
class softmax_layer: 
	def __init__(self, in_height, in_width, in_depth):
		# define input dimensions
		self.in_height = in_height
		self.in_width = in_width
		self.in_depth = in_depth
		
		# define output dimensions (they're the same, but needed for the next layer)
		self.out_height = in_height
		self.out_width = in_width
		self.out_depth = in_depth

	# a function that feeds the input volume through the network
	def forward(self, input_volume):
		# map e^x across the input volume
		exp_matrix = np.exp(input_volume.volume_slices)

		# normalize the matrix values - they now represent individual probabilities
		exp_matrix_sum = np.sum(exp_matrix) 
		exp_matrix = exp_matrix / exp_matrix_sum
		
		return volume(exp_matrix)