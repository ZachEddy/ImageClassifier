import numpy as np
from src import volume

class conv_layer:
	def __init__(self, field_size, filter_count, stride, pad):
		# set params as instance variables
		self.field_size = field_size
		self.filter_count = filter_count
		self.stride = stride
		self.pad = pad

	def feed_forward(self, input_volume):
		print self.pad
	

	# def generate_filters(self):
	# 	w = np.random.randn(n) * sqrt(2.0/n)