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
import time


class net_network:
	def __init__(self, layer_structure):
		# load the cifar-10 images and their corresponding labels
		cifar_data = load_cifar.images_to_volumes("image_data/data_batch_1")
		test_data = load_cifar.images_to_volumes("image_data/data_batch_2")
		self.image_volumes = cifar_data[0]
		self.image_labels = cifar_data[1]

		test_volumes = test_data[0]
		test_labels = test_data[1]

		# generate the layers based off of layer definitions from user
		self.layers = []
		self.build_layers(layer_structure)
		start = time.time()

		# print self.image_labels
		for i in range(len(self.image_volumes)):
			print i, "train"
			label = self.image_labels[i]
			self.forward(self.image_volumes[i])
			self.backward(label)
			self.train(0.01)
			# print self.layers[len(self.layers)-1].classify

		end = time.time()
		print end - start

		correct = 0.0
		test_amount = 1000
		for i in range(test_amount):
			self.forward(test_volumes[i])
			result = self.layers[len(self.layers)-1].classify
			print i, "test"
			if result == test_labels[i]:
				correct = correct + 1
		print (correct / test_amount) * 100
		
		
		# self.forward(self.image_volumes[0])

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
					previous.out_height,
					previous.out_width,
					previous.out_depth,
					layer['neuron_count'])
				self.layers.append(new_layer)
				previous = new_layer
			else:
				print "Unknown layer: \'%s\'" % (layer['type'])
				quit()
		# print self.layers

	def forward(self, image_volume):
		# go through each layer and compute the forward pass
		
		input_volume = self.layers[0].forward(image_volume)
		for layer in self.layers[1:]:
			input_volume = layer.forward(input_volume)
		np.sum(input_volume.volume_slices)


	def backward(self, label):
		flip_list = self.layers[::-1]

		output_with_gradients = flip_list[0].backward(label)
		for layer in flip_list[1:len(flip_list)-1]:
			layer.output_volume = output_with_gradients
			output_with_gradients = layer.backward()

	def train(self, rate):
		for layer in self.layers:
			layer.train(rate)
	# input volume should be an image volume
	def test(self, image_volume):
		

		input_volume = [
		[[1,2,2,1,2],
		[1,1,1,0,0],
		[0,0,0,2,0],
		[2,0,0,0,1],
		[0,2,2,2,1]],
		
		[[2,2,0,2,2],
		[1,2,0,2,2],
		[1,1,0,1,0],
		[2,0,1,0,2],
		[1,2,0,2,2]],
		
		[[2,2,1,0,2],
		[0,0,2,2,0],
		[1,0,1,2,1],
		[2,1,1,2,2],
		[0,0,2,2,1]]
		]
		input_volume = volume(np.array(input_volume) * 1.0)

		filter_one = [
		[[0,1,1],
		[1,0,1],
		[0,1,-1]],
		[[1,0,-1],
		[0,0,-1],
		[1,-1,-1]],
		[[0,1,0],
		[1,1,0],
		[-1,0,0]]
		]
		filter_one = volume(np.array(filter_one) * 1.0)

		filter_two = [
		[[-1,1,-1],
		[0,0,1],
		[-1,0,-1]],
		[[-1,1,1],
		[0,-1,0],
		[-1,-1,-1]],
		[[1,0,0],
		[-1,0,-1],
		[-1,1,1]]
		]
		filter_two = volume(np.array(filter_two) * 1.0)

		# def __init__(self, field_size, filter_count, stride, padding, in_height, in_width, in_depth):
		conv = conv_layer(3,2,2,1,5,5,3)
		filters = [filter_one, filter_two]
		conv.filters = filters
		conv.biases = np.array([1,0])

		output_volume = conv.forward(input_volume)
		print output_volume.volume_slices
		gradients = np.reshape(range(18), (2,3,3)) * 1.0
		gradients.fill(1)
		output_volume.gradient_slices = gradients


			
		conv.backward()

		print "Start"
		for f in conv.filters:
			print "volume and grads:"
			print f.gradient_slices
			print f.gradient_slices
		print "filter grads\n"
		print conv.input_volume.volume_slices, "inputs\n"
		print conv.input_volume.gradient_slices, "grad\n"



		# print output_volume.volume_slices
		# print output_volume.gradient_slices

		
		# print filter_one
		# # relu
		# # create the layer
		# relu = relu_layer(2,2,3)

		# # make the input volume
		# input_volume = np.array(range(-6,6))
		# input_volume = np.reshape(input_volume, (3,2,2))
		# input_volume = volume(input_volume)
		
		# # print the input volume
		# print input_volume.volume_slices, "1: input\n"
		
		# # make the output volume
		# output_volume = relu.forward(input_volume)
		# print output_volume.volume_slices, "2: output\n"

		# # make the output gradients, assign them to the output
		# output_gradients = np.array(range(-6,6)[::-1])
		# output_gradients = np.reshape(output_gradients, (3,2,2))

		# output_volume.gradient_slices = output_gradients
		# print output_gradients, "3: output grad\n"

		# relu.backward()

		# print relu.input_volume.gradient_slices, "4: input grad\n"
		# # print input_volume.gradient_slices
		# # relu



		# input_volume = np.array([2,3,4,5,1,2,3,4,3,6,8,4,3,6,7,0,5,3,6,3,7,2,1,7,9,6,0,5,7,4,3,5,6,7,5,8,0,6,3,4,2,5,7,0,2,10,5,7])
		# input_volume = np.reshape(input_volume, (3,4,4))
		# input_volume = volume(input_volume)
		# print input_volume.volume_slices

		# pool = pool_layer(2,4,4,3)
		# output_volume = pool.forward(input_volume)
		# pool.output_volume.gradient_slices.fill(1)


		# pool.backward()
		# print input_volume.gradient_slices
		# print output_volume.volume_slices
		# print "______________________________________________________ COL"
		# print pool.max_col_positions
		# print "______________________________________________________ ROW"
		# print pool.max_row_positions



		# fully con

		# input_volume = np.array([2,3,4,5,1,2,3,4])
		# input_volume = volume(np.reshape(input_volume, (2,2,2)))

		# weight_one = np.array([1,2,3,4,5,6,7,8])
		# weight_one = volume(np.reshape(weight_one, (2,2,2)))

		# weight_two = np.array([8,7,6,5,4,3,2,1])
		# weight_two = volume(np.reshape(weight_two, (2,2,2)))

		# weight_list = [weight_one, weight_two]
		# fully = fc_layer(2,2,2,2)
		# fully.weights = weight_list
		# fully.biases[0][0][1] = 2

		# print fully.forward(input_volume).volume_slices

		# fully.backward()
		# fully conn 
	
		# # here, bro
		# neurons = np.array([3.2, 5.1, -1.7])

		# neurons = np.reshape(neurons, (1,1,3))
		# print neurons
		# vol = volume(neurons)

		# softmax = softmax_layer(1,1,10)

		# softmax.forward(vol)
		# softmax.backward(0)

		# print softmax.input_volume.gradient_slices

		# return
		# # end here



		# input_volume = self.layers[0].forward(image_volume)
		# for layer in self.layers:
		# 	if type(layer) == "input_layer":			
		# 		continue
		
		# 	input_volume = layer.forward(input_volume)

		# print "Probabilites:", input_volume.volume_slices
