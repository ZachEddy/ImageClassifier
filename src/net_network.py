# a class made to manage feeding images to the network
from src.layers.conv_layer import conv_layer
from src.layers.pool_layer import pool_layer

class net_network:

	layers = []

	def __init__(self, layer_structure):
		self.build_layers(layer_structure)


	def build_layers(self, layer_structure):
		# build a layer based on what is provided in the 'initialization' module
		for layer in layer_structure:
			if layer['type'] == 'conv':
				self.layers.append(conv_layer(layer['field_size'], layer['filter_count'], layer['stride'], layer['padding']))
			elif layer['type'] == 'pool':
				self.layers.append(pool_layer(layer['field_size']))
			else:
				print "Unknown layer: \'%s\'" % (layer['type'])