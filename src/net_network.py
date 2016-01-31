# a class made to manage feeding images to the network
from src.layers.conv_layer import conv_layer
from src.layers.pool_layer import pool_layer
import load_cifar

class net_network:

	def __init__(self, layer_structure):
		self.layers = []
		self.build_layers(layer_structure)
		# load the cifar-10 images and their corresponding labels
		cifar_data = load_cifar.images_to_volumes()
		self.image_volumes = cifar_data[0]
		self.image_labels = cifar_data[1]

	def build_layers(self, layer_structure):
		# build a layer based on what is provided in the 'initialization' module
		for layer in layer_structure:
			if layer['type'] == 'conv':
				self.layers.append(conv_layer(layer['field_size'], layer['filter_count'], layer['stride'], layer['padding']))
			elif layer['type'] == 'pool':
				self.layers.append(pool_layer(layer['field_size']))
			else:
				print "Unknown layer: \'%s\'" % (layer['type'])