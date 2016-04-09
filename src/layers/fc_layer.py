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

		# setup the biases
		self.initialize_biases()

	# a function to initialize the weights of the fully connected layer
	def initalize_weights(self):
		weights = []
		weight_count = self.in_height * self.in_width * self.in_depth
		# initialize weights for each neuron in the network
		for i in range(self.out_width):
			weight_gen = np.random.randn(weight_count) * np.sqrt(2.0 / weight_count)
			weight_gen = np.reshape(weight_gen, (self.in_depth, self.in_height, self.in_width))
			weights.append(volume(weight_gen))
		# set as an instance variable for later methods
		self.weights = weights

	def initialize_biases(self):
		# make a matrix with the dimensions of the output volume
		biases = np.zeros((1,1,self.out_width))

		# fill the matrix with 0.1 (initial bias value) and make it an instance variable
		biases.fill(0.1)
		self.biases = volume(biases)

	# a function that feeds an input volume through the network
	def forward(self, input_volume):
		self.input_volume = input_volume
		output = np.array([])
		# each matrix of weights corresponds to a slice in the output volume
		
		# print self.biases.volume_slices[0][0]
		for weight in self.weights:
			output = np.append(output, np.sum(weight.volume_slices * input_volume.volume_slices))
		output = output + self.biases.volume_slices[0][0]

		# maybe redefine which axix the wights are on (height, width, or depth)? Something to consider
		self.output_volume = volume(np.reshape(output, (1,1, self.out_width)))
		
		return self.output_volume

	# a function that computes gradients for the fully connected layer
	def backward(self):

		# create a volume of zeros for the new gradient
		input_gradient = np.zeros((self.in_depth, self.in_height, self.in_width))

		for neuron_count in range(self.out_width):
			# find the gradient from the previous layer
			chain_gradient = self.output_volume.gradient_slices[0][0][neuron_count]

			# multiply the weight values by the previous gradient, then
			# add them to the current gradient
			input_gradient += self.weights[neuron_count].volume_slices * chain_gradient

			# create a new volume of zeros for the weight gradient 
			weight_gradient = np.zeros((self.in_depth, self.in_height, self.in_width))

			# calculate the weight gradient
			weight_gradient = self.input_volume.volume_slices * chain_gradient

			self.weights[neuron_count].gradient_slices = weight_gradient

		self.biases.gradient_slices = self.output_volume.gradient_slices
		self.input_volume.gradient_slices = input_gradient

		return self.input_volume


		# # zero-out existing gradients
		# self.input_volume.zero_gradient()
		# # self.input_volume.gradient_slices = add_padding(self.input_volume) #fix this
		# # loop through each group of weights, the corresponding bias, and the corresponding gradient from the previous layer
		# # it's an ugly zip function, but it makes everything else easier to read (also might be slow - something to think about)
		# # @TODO: you don't need bias in the zip
		# for weight, bias, chain in zip(self.weights, self.biases.volume_slices[0][0], self.output_volume.gradient_slices[0][0]):
		# 	weight.zero_gradient()
		# 	# compute the gradients of the input volume
		# 	self.input_volume.gradient_slices += weight.volume_slices * chain
		# 	# compute the gradients of each individual weight
		# 	weight.gradient_slices += self.input_volume.volume_slices * chain
		
		# # update the bias graidents
		# self.biases.zero_gradient()
		# self.biases.gradient_slices[0][0] += self.output_volume.gradient_slices[0][0]

		# # return the input volume with the new gradients
		# return self.input_volume

	def params_grads(self):
		aggregate = []
		aggregate.append({"params":self.biases.volume_slices, "grads":self.biases.gradient_slices, "instance":self})
		for i in range(len(self.weights)):
			aggregate.append({"params":self.weights[i].volume_slices, "grads":self.weights[i].gradient_slices, "instance":self})
		return aggregate

	def train(self, rate):
		for w in self.weights:
			w.volume_slices += -(w.gradient_slices * rate)
