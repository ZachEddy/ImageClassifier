# a class to manage the convolution layer
import numpy as np
from src import volume

class conv_layer:
	def __init__(self, field_size, filter_count, stride, pad):
		# set class variables
		self.field_size = field_size
		self.filter_count = filter_count
		self.stride = stride
		self.pad = pad

	def forward(self, input_volume):
		# I don't do anything yet
		return

	def backward(self, input_volume):
		# I don't do anything yet
		return