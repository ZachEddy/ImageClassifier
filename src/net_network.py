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
			self.layer_structure = self.load_structure()
			self.build_layers(self.layer_structure)
			self.load_params()
		else:
			self.layer_structure = layer_structure
			self.build_layers(self.layer_structure)
			self.initialize_params()

		






		# self.load_structure()
		# self.load_params(
		# if load_existing:
		# 	self.load_params()
		# else:
			
		# quit()
		# self.save_params()
		# start = time.time()
		

		# self.test(self.image_volumes[0])
		# quit()

		# self.forward(self.image_volumes[1])
		# print self.image_labels

		# # START
		# start = time.time()		
		# for i in range(1000):
		# 	print i, "train"
		# 	label = self.image_labels[i]
		# 	self.forward(self.image_volumes[i])
		# 	# print label, "label"
		# 	self.backward(label)
		# 	self.train(0.001)
		# # END
		# end = time.time()
		# print end - start

		# # START
		# correct = 0.0
		# test_amount = 1000
		# for i in range(test_amount):
		# 	self.forward(test_volumes[i])
		# 	result = self.layers[len(self.layers)-1].classify
		# 	print i, "test"
		# 	# print result
		# 	if result == test_labels[i]:
		# 		correct = correct + 1
		# print (correct / test_amount) * 100
		# # END
		
		# self.forward(self.image_volumes[0])
		# 
	




	# a function that initializes each layer depending on the user specs.
	# basically a massive if/elif/else block
	def build_layers(self, layer_structure):
		# build layers using the information provided from the user
		image_input = layer_structure.pop(0)

		# make sure the first alyer defined is input
		if image_input['type'] != 'input':
			print "First element in 'layer_structure' must be an input_layer type"
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


	def backward(self, label):
		output_with_grad = self.layers[len(self.layers)-1].backward(label)
		loss = self.layers[len(self.layers)-1].loss
		for i in range(1,len(self.layers)-1)[::-1]:
			self.layers[i].output_volume = output_with_grad
			output_with_grad = self.layers[i].backward()
		return loss

	def train(self, rate):
		for layer in self.layers:
			layer.train(rate)
	
	def params_grads(self):
		aggregate = []
		for layer_index in range(len(self.layers)):
			params_grads = self.layers[layer_index].params_grads()
			for param_grad_index in range(len(params_grads)):
				aggregate.append(params_grads[param_grad_index])
		return aggregate

	def initialize_params(self):
		for layer in self.layers:
			layer.initialize_params()

	def save_params(self):
		for layer in self.layers:
			layer.save_params(self.net_name)

	def save_structure(self):
		layer_file = open("saved_networks/"+self.net_name+"/structure.txt", "w")
		for layer_spec in self.layer_structure:
			layer_file.write(str(layer_spec)+"\n")
		layer_file.close()

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

	def classify(self, image_volume):
		self.forward(image_volume)
		result = self.layers[len(self.layers)-1].classify
		return result















	def test(self, image_volume):


		# define input volume
		input_volume = np.array([1,2,0,2,0,2,1,1,0,2,1,0,0,1,1,2,2,1,0,2,2,1,0,2,2,0,0,1,2,2,2,0,2,0,1,1,1,1,0,2,0,0,2,2,1,1,1,0,1,0]) * 1.0
		input_volume = np.reshape(input_volume, (2,5,5))
		input_volume = volume(input_volume)

		print input_volume.volume_slices
		# define filters
		filter_one = np.array([3,-5,6,2,-2,-1,0,-2]) * 1.0
		filter_one = np.reshape(filter_one, (2,2,2))
		filter_one = volume(filter_one)

		filter_two = np.array([1,0,3,-4,-5,6,-1,-2]) * 1.0
		filter_two = np.reshape(filter_two, (2,2,2))
		filter_two = volume(filter_two)

		filters = [filter_one, filter_two]


		# define new conv layer with zero bias and filters
		conv = conv_layer(2,2,1,0,5,5,2)
		conv.filters = filters
		conv.biases.volume_slices.fill(0.0)

		conv_volume = conv.forward(input_volume)

		print "Convolution Output"
		print conv_volume.volume_slices,"\n"

		relu_one = relu_layer(4,4,2)
		relu_one_volume = relu_one.forward(conv_volume)

		print "Relu Output"
		print relu_one_volume.volume_slices, "\n"

		pool = pool_layer(2,4,4,2)
		pool_volume = pool.forward(conv_volume)

		print "Pool Output"
		print pool_volume.volume_slices, "\n"

		# print pool.max_row_positions, "max-row"
		# print pool.max_col_positions, "max-col"

		fc = fc_layer(2,2,2,3)
			
		weight_one = np.zeros((1,1,8))
		weight_one = np.reshape(weight_one,(2,2,2))
		weight_one.fill(1.4)
		# weight_one[0][0][0] = 100
		weight_one = volume(weight_one)

		weight_two = np.zeros((1,1,8))
		weight_two = np.reshape(weight_two,(2,2,2))
		weight_two.fill(1.5)
		weight_two = volume(weight_two)

		weight_three = np.zeros((1,1,8))
		weight_three = np.reshape(weight_three,(2,2,2))
		weight_three.fill(1.6)
		weight_three = volume(weight_three)

		weights = [weight_one, weight_two, weight_three]

		fc.weights = weights
		fc.biases.volume_slices.fill(0)
		fc_volume = fc.forward(pool_volume)
		print "Fully Connected Output"
		print fc_volume.volume_slices, "\n"
		
		soft = softmax_layer(1,1,3)


		soft_volume = soft.forward(fc_volume)
		
		print "Softmax Output"
		print soft_volume.volume_slices, "\n"


		soft_backward = soft.backward(1)
		print soft.input_volume.volume_slices, "test"
		print fc.output_volume.volume_slices, "test"
		print fc.output_volume.gradient_slices, "test"

		print fc.output_volume.gradient_slices

		fc_backward = fc.backward()
		print fc.input_volume.gradient_slices, "voici"
		
		print pool.output_volume.gradient_slices

		print "\nPool Backward"
		pool_backward = pool.backward()

		print pool_backward.gradient_slices


		print "\nRelu Backward"

		relu_one.output_volume = pool_backward
		relu_backward = relu_one.backward()

		print relu_backward.gradient_slices

		print "Convolution Backward"
		conv.output_volume = relu_backward
		conv_backward = conv.backward()
		print conv.input_volume.gradient_slices
		for f in conv.filters:
			print "Filter:"
			print f.gradient_slices



		

		# print fc_backward.gradient_slices








		# input_volume = [
		# [[1,8,2,0,2,1],
		# ],
		
		# [[2,2,0,2,2,1],
		# ],
		
		# [[2,2,1,0,2,2],
		# ]
		# ]

		# input_volume = volume(np.array(input_volume) * 1.0)	
		# print np.sum(input_volume.volume_slices), "fsd"
		# fc = fc_layer(1,6,3,2)
		# for i in fc.weights:
		# 	i.volume_slices.fill(1)

		# output_volume = fc.forward(input_volume)
		# output_volume.gradient_slices.fill(1)
		# fc.backward()
		# print "Here, motherfuckers"
		# for i in fc.params_grads():
		# 	print "p'rams grads"
		# 	print i["params"], "param"
		# 	print i["grads"], "grad"

		# print output_volume.volume_slices
		# print input_volume.volume_slices, "INPUT\n"
		# print input_volume.gradient_slices

		# # for i in fc.weights:
		# # 	print i.gradient_slices
		# # # print fc.weights.volume_slices



		# # # pool = pool_layer(2,6,6,3)
		# # # output = pool.forward(input_volume)
		# # # print output.volume_slices
		# # # print pool.max_row_positions, "row\n"
		# # # print pool.max_col_positions, "col\n"
		# # # for i, j in zip(pool.max_row_positions.flatten(), pool.max_col_positions.flatten()):
		# # # 	# print i,j
		# # # 	print input_volume.volume_slices[0][i][j]

		# return
		# input_volume = [
		# [[1,2,2,1,2],
		# [1,1,1,0,0],
		# [0,0,0,2,0],
		# [2,0,0,0,1],
		# [0,2,2,2,1]],
		
		# [[2,2,0,2,2],
		# [1,2,0,2,2],
		# [1,1,0,1,0],
		# [2,0,1,0,2],
		# [1,2,0,2,2]],
		
		# [[2,2,1,0,2],
		# [0,0,2,2,0],
		# [1,0,1,2,1],
		# [2,1,1,2,2],
		# [0,0,2,2,1]]
		# ]
		# input_volume = volume(np.array(input_volume) * 1.0)

		# filter_one = [
		# [[0,1,1],
		# [1,0,1],
		# [0,1,-1]],
		# [[1,0,-1],
		# [0,0,-1],
		# [1,-1,-1]],
		# [[0,1,0],
		# [1,1,0],
		# [-1,0,0]]
		# ]
		# filter_one = volume(np.array(filter_one) * 1.0)

		# filter_two = [
		# [[-1,1,-1],
		# [0,0,1],
		# [-1,0,-1]],
		# [[-1,1,1],
		# [0,-1,0],
		# [-1,-1,-1]],
		# [[1,0,0],
		# [-1,0,-1],
		# [-1,1,1]]
		# ]
		# filter_two = volume(np.array(filter_two) * 1.0)

		# # def __init__(self, field_size, filter_count, stride, padding, in_height, in_width, in_depth):
		# conv = conv_layer(3,2,2,1,5,5,3)
		# filters = [filter_one, filter_two]
		# conv.filters = filters
		# conv.biases = volume(np.array([[[1,0]]]))

		# output_volume = conv.forward(input_volume)
		# print output_volume.volume_slices
		# gradients = np.reshape(range(18), (2,3,3)) * 1.0
		# gradients.fill(1)
		# output_volume.gradient_slices = gradients

		# # conv.
			
		# conv.backward()
		# print "starting here, motherfuckers"
		# for i in conv.params_grads():
		# 	print
		# 	print i["params"]
		# 	print i["grads"]

		# # print "Start"
		# # for f in conv.filters:
		# # 	print "volume and grads:"
		# # 	print f.gradient_slices
		# # 	print f.gradient_slices
		# # print "filter grads\n"
		# # print conv.input_volume.volume_slices, "inputs\n"
		# # print conv.input_volume.gradient_slices, "grad\n"



		# # # print output_volume.volume_slices
		# # # print output_volume.gradient_slices

		
		# # # print filter_one
		# # # # relu
		# # # # create the layer
		# # # relu = relu_layer(2,2,3)

		# # # # make the input volume
		# # # input_volume = np.array(range(-6,6))
		# # # input_volume = np.reshape(input_volume, (3,2,2))
		# # # input_volume = volume(input_volume)
		
		# # # # print the input volume
		# # # print input_volume.volume_slices, "1: input\n"
		
		# # # # make the output volume
		# # # output_volume = relu.forward(input_volume)
		# # # print output_volume.volume_slices, "2: output\n"

		# # # # make the output gradients, assign them to the output
		# # # output_gradients = np.array(range(-6,6)[::-1])
		# # # output_gradients = np.reshape(output_gradients, (3,2,2))

		# # # output_volume.gradient_slices = output_gradients
		# # # print output_gradients, "3: output grad\n"

		# # # relu.backward()

		# # # print relu.input_volume.gradient_slices, "4: input grad\n"
		# # # # print input_volume.gradient_slices
		# # # # relu



		# # # input_volume = np.array([2,3,4,5,1,2,3,4,3,6,8,4,3,6,7,0,5,3,6,3,7,2,1,7,9,6,0,5,7,4,3,5,6,7,5,8,0,6,3,4,2,5,7,0,2,10,5,7])
		# # # input_volume = np.reshape(input_volume, (3,4,4))
		# # # input_volume = volume(input_volume)
		# # # print input_volume.volume_slices

		# # # pool = pool_layer(2,4,4,3)
		# # # output_volume = pool.forward(input_volume)
		# # # pool.output_volume.gradient_slices.fill(1)


		# # # pool.backward()
		# # # print input_volume.gradient_slices
		# # # print output_volume.volume_slices
		# # # print "______________________________________________________ COL"
		# # # print pool.max_col_positions
		# # # print "______________________________________________________ ROW"
		# # # print pool.max_row_positions



		# # # fully con

		# # # input_volume = np.array([2,3,4,5,1,2,3,4])
		# # # input_volume = volume(np.reshape(input_volume, (2,2,2)))

		# # # weight_one = np.array([1,2,3,4,5,6,7,8])
		# # # weight_one = volume(np.reshape(weight_one, (2,2,2)))

		# # # weight_two = np.array([8,7,6,5,4,3,2,1])
		# # # weight_two = volume(np.reshape(weight_two, (2,2,2)))

		# # # weight_list = [weight_one, weight_two]
		# # # fully = fc_layer(2,2,2,2)
		# # # fully.weights = weight_list
		# # # fully.biases[0][0][1] = 2

		# # # print fully.forward(input_volume).volume_slices

		# # # fully.backward()
		# # # fully conn 
	
		# # # # here, bro
		# # # neurons = np.array([3.2, 5.1, -1.7])

		# # # neurons = np.reshape(neurons, (1,1,3))
		# # # print neurons
		# # # vol = volume(neurons)

		# # # softmax = softmax_layer(1,1,10)

		# # # softmax.forward(vol)
		# # # softmax.backward(0)

		# # # print softmax.input_volume.gradient_slices

		# # # return
		# # # # end here



		# # # input_volume = self.layers[0].forward(image_volume)
		# # # for layer in self.layers:
		# # # 	if type(layer) == "input_layer":			
		# # # 		continue
		
		# # # 	input_volume = layer.forward(input_volume)

		# # # print "Probabilites:", input_volume.volume_slices
