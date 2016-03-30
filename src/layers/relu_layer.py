import numpy as np
from src.volume import volume
# a class to manage the ReLu activation layer
class relu_layer: 
	def __init__(self, in_height, in_width, in_depth):
		# define input dimensions
		self.in_height = in_height
		self.in_width = in_width
		self.in_depth = in_depth
		
		# define output dimensions (they're the same, but needed for the next layer)
		self.out_height = in_height
		self.out_width = in_width
		self.out_depth = in_depth

		# vectorize the ReLU activation function so it can easily be mapped over neuron volumes
		self.activate_volume = np.vectorize(self.activate_single)

	# a function that values the ReLu activation function on a single neuron in the volume
	def activate_single(self, neuron):
		# the activation function is simply max(0,x). consider making it leaky at some point down the road
		return max(0, neuron)

	# a function that feeds the input volume through the network
	def forward(self, input_volume):
		# save volumes for backprop
		self.input_volume = input_volume
		self.output_volume = volume(self.activate_volume(input_volume.volume_slices))

		return self.output_volume

	# apply the max(0,x) thresholding (ReLu activation) across the gradients
	def backward(self):
		# flatten the output gradients and ouput neuron values to make iteration easier
		output_values = self.output_volume.volume_slices.flatten()
		output_gradient = self.output_volume.gradient_slices.flatten()
		# merge output gradients and output neuron values for backprop
		values = zip(output_values, output_gradient)
		# iteratively find gradients by appending to an empty list
		input_gradient = []

		# go through gradient and output volumes
		for output_value, output_gradient in values:
			# do chain gradient from output gradient to input gradient
			if output_value <= 0:
				input_gradient.append(0)
			else:
				input_gradient.append(output_gradient)

		# reshape the gradients and add them to the input volume's gradient
		self.input_volume.gradient_slices = np.reshape(input_gradient, (self.in_depth, self.in_height, self.in_width))
		return self.input_volume

	def train(self, rate):
		return