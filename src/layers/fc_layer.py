import numpy as np
from src.volume import volume
# a class to manage the ReLu activation layer
class fc_layer: 
	def __init__(self, name, in_height, in_width, in_depth, neuron_count):
		# set name for weight saving/loading purposes
		self.name = name

		# define input dimensions
		self.in_height = in_height
		self.in_width = in_width
		self.in_depth = in_depth

		# define output dimensions
		self.out_height = 1
		self.out_width = neuron_count
		self.out_depth = 1

	# a function to initialize the weights of the fully connected layer
	def initialize_weights(self):
		weights = []
		weight_count = self.in_height * self.in_width * self.in_depth
		# initialize weights for each neuron in the network
		for i in range(self.out_width):
			weight_gen = np.random.randn(weight_count) * np.sqrt(2.0 / weight_count)
			weight_gen = np.reshape(weight_gen, (self.in_depth, self.in_height, self.in_width))
			weights.append(volume(weight_gen))
		# set as an instance variable for later methods
		self.weights = weights

	# a function to initialize the network biases
	def initialize_biases(self):
		# make a matrix with the dimensions of the output volume
		biases = np.zeros((1,1,self.out_width))

		# fill the matrix with 0.1 (initial bias value) and make it an instance variable
		biases.fill(0.1)
		self.biases = volume(biases)

	# a function to initialize all network parameters (weights and biases)
	def initialize_params(self):
		self.initialize_biases()
		self.initialize_weights()

	# a function to save weights and biases
	def save_params(self, net_name):
		weight_params = []
		for weight in self.weights:
			weight_params.append(weight.volume_slices)
		np.savez("saved_networks/"+net_name+"/"+self.name, biases = self.biases.volume_slices, weights = weight_params)

	# a function to load existing weights from a file
	def load_params(self, net_name):
		params = np.load("saved_networks/"+net_name+"/"+self.name + ".npz")
		weights = []
		for weight_values in params["weights"]:
			weights.append(volume(weight_values))
		self.weights = weights
		self.biases = volume(params["biases"])
	
	# a function that feeds an input volume through the network
	def forward(self, input_volume):
		self.input_volume = input_volume
		
		output = np.array([])
		# each matrix of weights corresponds to a slice in the output volume
		for weight in self.weights:
			# print weight.volume_slices
			output = np.append(output, np.sum(weight.volume_slices * input_volume.volume_slices))
		output = output + self.biases.volume_slices[0][0]

		# maybe redefine which axix the wights are on (height, width, or depth)? Something to consider
		self.output_volume = volume(np.reshape(output, (1,1, self.out_width)))
		return self.output_volume

	# backpropagation for fully connected layer
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

	# a method to return all network parameters and their gradients for training
	def params_grads(self):
		aggregate = []
		aggregate.append({"params":self.biases.volume_slices, "grads":self.biases.gradient_slices, "instance":self})
		for i in range(len(self.weights)):
			aggregate.append({"params":self.weights[i].volume_slices, "grads":self.weights[i].gradient_slices, "instance":self})
		return aggregate

	def train(self, rate):
		for w in self.weights:
			w.volume_slices += -(w.gradient_slices * rate)