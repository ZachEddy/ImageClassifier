# a class made to manage feeding images to the network
import load_cifar
import numpy as np
from src.layers.conv_layer import conv_layer
from src.layers.pool_layer import pool_layer
from src.layers.input_layer import input_layer
from src.layers.relu_layer import relu_layer
from src.layers.fc_layer import fc_layer
from src.layers.tanh_layer import tanh_layer
from src.volume import volume

class net_network:
	def __init__(self, layer_structure):
		# load the cifar-10 images and their corresponding labels
		cifar_data = load_cifar.images_to_volumes()
		self.image_volumes = cifar_data[0]
		self.image_labels = cifar_data[1]

		# generate the layers based off of layer definitions from user
		self.layers = []
		self.build_layers(layer_structure)
		self.forward(self.image_volumes[0])

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

	# input volume should be an image volume
	def forward(self, image_volume):

		print "START!"
		# print image_volume.volume_slices
		# print "_--__-_---------~~~~~~~~~~~~~~"
		input_volume = self.layers[0].forward(image_volume)
		for layer in self.layers:
			if type(layer) == "input_layer":			
				continue

			# print input_volume.volume_slices
			# print layer, "_______________________________________________________________"
			# print ""
			input_volume = layer.forward(input_volume)

		print "DONE!"

		print input_volume.volume_slices

		# vol = np.array([[[1,2],[3,4]],[[4,5],[6,7]]])
		# print "vol", vol
		# # def __init__(self, in_height, in_width, in_depth, neuron_count):
		# fc = fc_layer(2,2,2,2)
		# vol = volume(vol)

		# vol = fc.forward(vol)

		# print "fc", vol.volume_slices

		# tanh = tanh_layer(1,1,2)

		# vol = tanh.forward(vol)
		# print "tanh", vol.volume_slices
		# print image_volume.volume_slices
		# first_vol = self.layers[0].forward(image_volume)

		# # print first_vol.volume_slices

		# second_vol = self.layers[1].forward(first_vol)
		# # print second_vol.volume_slices
		# # ignores the relu layer
		# third_vol = self.layers[3].forward(second_vol)
		# print third_vol.volume_slices




		# # testing the pool layer
		# test_volume = random.sample(range(-100,100), 192)
		# test_volume = np.reshape(test_volume, (3,8,8))
		# test_volume = volume(test_volume)



		# print test_volume.volume_slices
		# relu_test = relu_layer(3,8,8)
		# test_vol = relu_test.forward(test_volume)
		# print "-----"
		# print test_vol.volume_slices



		
		# print self.layers[1].forward(input_volume)





		# Tons of commented code that I don't want to delete. I'm sure it'll never be useful again... but who knows!?

		# # print data

		# # print f 
		# # quit()
		# # data[256, 256] = [255, 0, 0]
		# # img = Image.fromarray(data, 'RGB')
		# # img.save('my.png')
		# # 
		# rgb = []
		# for i in range(input_volume.height):
		# 	f = []
		# 	for j in range(input_volume.width):
		# 		f.append([input_volume.volume_slices[0][i][j], input_volume.volume_slices[1][i][j], input_volume.volume_slices[2][i][j]])
		# 	rgb.append(f)



		# # # print len(input_volume.volume_slices[0][0])
		# # rgb = np.array(input_volume.volume_slices, dtype=np.uint8)
		# img = Image.fromarray(np.array(rgb), 'RGB')
		# # print data

		# # img = Image.fromarray(rgb, 'RGB')
		# img.save('my.png')

		# # # self.layers[1].forward(input_volume)




		# t = self.layers[1]
		# t.in_height = 7
		# t.in_width = 7
		# t.in_depth = 3
		# t.field_size = 3
		# t.stride = 2
		# t.filter_count = 2
		# t.calc_output_dimensions()

		# print t.out_depth, t.out_width, t.out_height


		# input_vol = [0,0,0,0,0,0,0,0,0,1,0,2,1,0,0,2,2,1,1,0,0,0,2,2,0,1,0,0,0,0,0,0,1,0,0,0,1,1,2,1,0,0,0,0,0,0,0,0,0,
		# 	0,0,0,0,0,0,0,0,2,2,1,1,1,0,0,2,0,0,2,1,0,0,1,2,1,2,0,0,0,1,1,0,0,0,0,0,1,0,0,2,0,0,0,0,0,0,0,0,0,
		# 	0,0,0,0,0,0,0,0,2,0,2,0,2,0,0,0,1,1,0,0,0,0,2,1,0,0,2,0,0,1,0,2,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]

		# input_vol = np.reshape(input_vol, (3,7,7))
		
		
		# # print input_vol[0]
		# for i in range(len(input_vol)):
		# 	input_vol[i] = np.transpose(input_vol[i])

		# # print input_vol

		# # f_1 = [1,0,0,1,0,-1,0,-1,-1,0,-1,-1,-1,1,0,1,0,-1,-1,-1,1,-1,1,1,1,-1,-1]
		# # f_2 = [1,1,0,1,1,-1,-1,0,-1,1,0,0,1,0,1,1,1,-1,0,1,1,-1,0,0,0,1,-1]
		
		# # f_2 = [1,1,-1,1,1,0,0,-1,-1,1,1,1,0,0,1,0,1,-1,0,-1,0,1,0,1,1,0,-1]
		# f_1 = np.reshape(f_1,(3,3,3))
		# f_2 = np.reshape(f_2,(3,3,3))

		# # f_1 = np.transpose(f_1)
		# # f_2 = np.transpose(f_2)

		# f_1 = volume(f_1)
		# f_2 = volume(f_2)

		# f = [f_1, f_2]
		# print f_1.volume_slices

		# t.filters = f
		
		# print t.forward(volume(input_vol))


		# # self.field_size = field_size
		# # self.filter_count = filter_count
		# # self.stride = stride
		# # self.padding = padding

		# # # specify input dimensions
		# # self.in_height = in_height
		# # self.in_width = in_width
		# # self.in_depth = in_depth

		# # print input_volume.volume_slices[:,0,0]
		# # quit()
		# # self.layers[1].forward(input_volume)

		# return 
		# # self.layers[1].forward(np.array(input_volume))
		# # current_volume = input_volume
		# # for layer in self.layers:
