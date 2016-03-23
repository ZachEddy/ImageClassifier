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
		# subtract the max output from all output values to avoid exponential explosion
		max_input = np.max(input_volume.volume_slices)
		exp_matrix = input_volume.volume_slices - max_input

		# map e^x across list to find unnormalized probabilities. save them for backpropagation
		exp_matrix = np.exp(exp_matrix)
		# self.exp = exp_matrix

		# normalize the probabilities to be between 0 and 1
		exp_matrix_sum = np.sum(exp_matrix)
		exp_matrix = exp_matrix / exp_matrix_sum

		# save volumes for backprop
		self.input_volume = input_volume
		self.output_volume = volume(exp_matrix)
		return self.output_volume

	# a function to find gradients of the softmax layer
	def backward(self, label):
		# zero out existing gradients
		self.input_volume.zero_gradient()
		for i in range(len(self.output_volume.volume_slices[0,0,:])):
			# check if output corresponds probability to the actual label (only something we know during training)
			# if an output doesn't match the label, it will be zero
			label_hit = 1 if i == label else 0
			
			# find gradient by subtracting the probability from the label hit (either one or zero)
			self.input_volume.gradient_slices[0,0,i] = -(label_hit - self.output_volume.volume_slices[0,0,i])
		
		# evaluate the loss and store it as an instance variable
		self.loss = -np.log10(self.output_volume.volume_slices[0,0,label])
		return self.input_volume
