# a class made to manage feeding images to the network
import load_cifar
from src.layers.conv_layer import conv_layer
from src.layers.pool_layer import pool_layer
from src.layers.input_layer import input_layer
from src.layers.relu_layer import relu_layer

class net_network:

	def __init__(self, layer_structure):
		# generate the layers based off of layer definitions from user
		self.layers = []
		self.build_layers(layer_structure)

		# load the cifar-10 images and their corresponding labels
		cifar_data = load_cifar.images_to_volumes()
		self.image_volumes = cifar_data[0]
		self.image_labels = cifar_data[1]

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

	# input volume should be an image
	def forward(self, input_volume):
		current_volume = input_volume
		# for layer in self.layers:
