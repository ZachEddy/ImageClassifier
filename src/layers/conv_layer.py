# a class to manage the convolution layer
import numpy as np
from src import volume
from src.net_utilities import is_int
	
class conv_layer:
	def __init__(self, field_size, filter_count, stride, padding, in_height, in_width, in_depth):
		# set class variables
		self.field_size = field_size
		self.filter_count = filter_count
		self.stride = stride
		self.padding = padding

		# specify input dimensions
		self.in_height = in_height
		self.in_width = in_width
		self.in_depth = in_depth

		# calculate output dimensions based on input dimensions
		self.calc_output_dimensions()

	def calc_output_dimensions(self):
		# define the output volume dimensions and make sure the inputs are valid
		self.out_height = (self.in_height - self.field_size + 2.0 * self.padding) / self.stride + 1
		self.out_width = (self.in_width - self.field_size + 2.0 * self.padding) / self.stride + 1
		self.out_depth = self.filter_count
		
		# ensure integer dimensions for output volume
		if not(is_int(self.out_height)) or (not is_int(self.out_width)) or (not is_int(self.out_depth)):
			print "Input dimensions into conv layer aren't valid"
			exit()
			
	def forward(self, input_volume):
		# I don't do anything yet
		return

	def backward(self, input_volume):
		# I don't do anything yet
		return