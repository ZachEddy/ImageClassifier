# a class made to manage feeding images to the network

import load_cifar
import numpy as np
from src.layers.conv_layer import conv_layer
from src.layers.pool_layer import pool_layer
from src.layers.input_layer import input_layer
from src.layers.relu_layer import relu_layer
from src.layers.fc_layer import fc_layer
from src.layers.tanh_layer import tanh_layer
from src.layers.softmax_layer import softmax_layer
from src.volume import volume

class net_network:
	def __init__(self, layer_structure, net_name):
		# load the cifar-10 images and their corresponding labels
		cifar_data = load_cifar.images_to_volumes("image_data/data_batch_1")
		test_data = load_cifar.images_to_volumes("image_data/data_batch_2")
		self.image_volumes = cifar_data[0]
		self.image_labels = cifar_data[1]

		test_volumes = test_data[0]
		test_labels = test_data[1]
		self.net_name = net_name
		# generate the layers based off of layer definitions from user
		self.layers = []

		if layer_structure == None:
			print "~~ Loading network '%s' from saved_networks directory..." % (net_name)
			self.pretrained = True
			self.layer_structure = self.load_structure()
			self.build_layers(self.layer_structure)
			self.load_params()
			print "~~ Done!"
		else:
			print "~~ Initializing untrained network..."
			self.pretrained = False
			self.layer_structure = layer_structure
			self.build_layers(self.layer_structure)
			self.initialize_params()
			print "~~ Done!"
		print

	# a function that initializes each layer depending on the user specs.
	# basically a massive if/elif/else block
	def build_layers(self, layer_structure):
		# build layers using the information provided from the user
		image_input = layer_structure.pop(0)

		# make sure the first alyer defined is input
		if image_input['type'] != 'input':
			print "~~ First element in 'layer_structure' must be an input_layer type"
			exit()

		# add a new layer to the global list of layers
		self.layers.append(input_layer(image_input['out_height'], image_input['out_width'], image_input['out_depth']))		
		
		# have a reference to the previous layer because it's required when building the following layer
		previous = self.layers[0]

		# iterate through all the user-definitions and build the corresponding layer
		for layer in layer_structure:
			if layer['type'] == 'conv':
				# create the new conv layer
				new_layer = conv_layer(
					layer['name'],
					layer['field_size'],
					layer['filter_count'], 
					layer['stride'], 
					layer['padding'], 
					previous.out_height, 
					previous.out_width, 
					previous.out_depth)

				# add the new conv layer (dupe code)
				self.layers.append(new_layer)
				# change the previous layer to the current (so the next layer can talk to it)
				previous = new_layer
			elif layer['type'] == 'pool':
				# create the new poolng layer
				new_layer = pool_layer(
					layer['field_size'],
					previous.out_height,
					previous.out_width, 
					previous.out_depth)
				
				# add the new pooling layer (dupe code)
				self.layers.append(new_layer)
				# change the previous layer to the current (so the next layer can talk to it)
				previous = new_layer
			elif layer['type'] == 'relu':
				# create new ReLu layer (activation layer)
				new_layer = relu_layer(
					previous.out_height,
					previous.out_width,
					previous.out_depth)
				# add the new conv layer (dupe code)
				self.layers.append(new_layer)
				previous = new_layer
			elif layer['type'] == 'tanh':
				# create new ReLu layer (activation layer)
				new_layer = tanh_layer(
					previous.out_height,
					previous.out_width,
					previous.out_depth)
				self.layers.append(new_layer)
				previous = new_layer
				# add the new conv layer (dupe code)
			elif layer['type'] == 'softmax':
				new_layer = softmax_layer(
					previous.out_height,
					previous.out_width,
					previous.out_depth)
				self.layers.append(new_layer)
				previous = new_layer
			elif layer['type'] == 'fully_connected':
				# fully connected layer
				new_layer = fc_layer(
					layer['name'],
					previous.out_height,
					previous.out_width,
					previous.out_depth,
					layer['neuron_count'])
				self.layers.append(new_layer)
				previous = new_layer
			else:
				print "Unknown layer: \'%s\'" % (layer['type'])
				quit()
		layer_structure.insert(0, image_input)
		# print self.layers

	# a function to feed an image through the network and classify
	def forward(self, image_volume):
		# go through each layer and compute the forward pass
		input_volume = self.layers[0].forward(image_volume)
		# print input_volume.volume_slices
		for layer in self.layers[1:]:
			# print layer, "\n\n\n"

			# start = time.time()
			# print len(input_volume.volume_slices[0]), "f_height"
			# print len(input_volume.volume_slices[0][0]), "f_width"
			input_volume = layer.forward(input_volume)
			# print input_volume.volume_slices
			# end = time.time()
			# print end - start, layer
		# print "END:", time.time() - top

	# go through each layer and find gradients
	def backward(self, label):
		output_with_grad = self.layers[len(self.layers)-1].backward(label)
		loss = self.layers[len(self.layers)-1].loss
		for i in range(1,len(self.layers)-1)[::-1]:
			self.layers[i].output_volume = output_with_grad
			output_with_grad = self.layers[i].backward()
		return loss
	
	def params_grads(self):
		aggregate = []
		for layer_index in range(len(self.layers)):
			params_grads = self.layers[layer_index].params_grads()
			for param_grad_index in range(len(params_grads)):
				aggregate.append(params_grads[param_grad_index])
		return aggregate

	# go through each layer and initialize weights, biases, etc.
	def initialize_params(self):
		for layer in self.layers:
			layer.initialize_params()

	# a function to save all the weights, biases, etc. 
	def save_params(self):
		for layer in self.layers:
			layer.save_params(self.net_name)

	# a function to save the layer specs as a .txt file
	def save_structure(self):
		layer_file = open("saved_networks/"+self.net_name+"/structure.txt", "w")
		for layer_spec in self.layer_structure:
			layer_file.write(str(layer_spec)+"\n")
		layer_file.close()

	# a function that saves the entire network
	def save_network(self):
		import os
		directory = "saved_networks/"+self.net_name
		if not os.path.exists(directory):
			os.makedirs(directory)
		self.save_structure()
		self.save_params()

	# a function to load all network weights and biases
	def load_params(self):
		for layer in self.layers:
			layer.load_params(self.net_name)

	# a function to load network layer definitions
	def load_structure(self):
		import ast
		layer_file = open("saved_networks/"+self.net_name+"/structure.txt")
		layer_structure = []
		for line in layer_file:
			layer_dict = ast.literal_eval(line)
			layer_structure.append(layer_dict)
		return layer_structure

	# a function that loads the entire network in a single call
	def load_network(self):
		self.layer_structure = self.load_structure()
		self.load_params()

	# a method to feed an image through the network and return the result
	def classify(self, image_volume):
		self.forward(image_volume)
		result = self.layers[len(self.layers)-1].classify
		return result
