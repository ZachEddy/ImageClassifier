# a class made to manage feeding images to the network
import load_cifar
import numpy as np
from src.layers.conv_layer import conv_layer
from src.layers.pool_layer import pool_layer
from src.layers.input_layer import input_layer
from src.layers.relu_layer import relu_layer
from src.volume import volume
from PIL import Image

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
			else:
				print "Unknown layer: \'%s\'" % (layer['type'])
		print self.layers

	# input volume should be an image volume
	def forward(self, input_volume):
		# testing the convolution layer
		print self.layers[1].forward(input_volume)





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
