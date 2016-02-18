import numpy as np
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

	def relu_single(neuron):
		return max(0, neuron)

	relu_volume = np.vectorize(relu_single)
