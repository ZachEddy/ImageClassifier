# a class to manage the convolution layer
import numpy as np
from src.volume import volume
from src.net_utilities import is_int
	
class conv_layer:
	# constructor
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

		# initialize filter volumes
		self.initialize_filters()

	def add_padding(self, input_volume):
		# minimize number of references to instance variable 'self.padding'
		padding = self.padding
		if padding > 0:
			pad_specs = ((0,0), (padding, padding), (padding, padding))
			input_volume.volume_slices = np.pad(input_volume.volume_slices, pad_specs, mode='constant',constant_values=0)
		return input_volume


	# a function to calculate output volume height based on input volume height
	def calc_output_height(self):
		return (self.in_height - self.field_size + 2.0 * self.padding) / self.stride + 1.0

	# a function to calculate output volume width based on input volume width
	def calc_output_width(self):
		return (self.in_width - self.field_size + 2.0 * self.padding) / self.stride + 1.0

	# a function to check user-defined layer specs will produce a valid output (no non-integer pixel values)
	def check_user_definition(self):
		# quit the program if any of the dimensions aren't integers
		print self.out_height, self.out_width, self.out_depth
		if not(is_int(self.out_height)) or (not is_int(self.out_width)) or (not is_int(self.out_depth)):
			print "Input dimensions into conv layer aren't valid"
			quit()

	# a function to make sure that inputs into the conv layer match with what it expects
	def check_input_dimensions(self, input_volume):
		if (input_volume.height != self.in_height) or (input_volume.width != self.in_width) or (input_volume.depth != self.in_depth):
			print "Dimensions of input volume do not match expected dimensions"
			quit()
	
	# a function to calculate the dimensions of the output volume
	def calc_output_dimensions(self):
		# define the output volume dimensions and make sure the inputs are valid
		self.out_height = self.calc_output_height()
		self.out_width = self.calc_output_width()
		self.out_depth = self.filter_count

		# check to make sure the user enterered valid layer specs (no part-pixels)
		self.check_user_definition()
		
		# assuming the output volumes are integer values, then make them integers in computer terms (ex: 6.0 --> 6) 
		self.out_height = int(self.out_height)
		self.out_width = int(self.out_width)
		self.out_depth = int(self.out_depth)

	# a function to initialize the filter volumes
	def initialize_filters(self):
		# a list to hold all the filter volumes - depth of volumes is the depth of the input volume
		# a filter will be paired with inputs from each layer of depth
		filters = []

		# calculate the area of the receptive field
		weight_count = (self.field_size ** 2) * self.in_depth

		# create filter volumes for the number of sets the user specifies. If depth is 3, and the user specifies
		# 20 filter sets, then there will be 60 total filter. If the receptive field size is 3, then the 
		# total number of weights will be 540. (3 * 20 * (3^2))
		for i in range(self.filter_count):
			# initialize weights for every filter
			# recommended initalization function for ReLu: np.random.randn(n) * np.sqrt(2.0 / n)
			filter_volume = np.random.randn(weight_count) * np.sqrt(2.0 / weight_count)
			filter_volume = np.reshape(filter_volume, (self.in_depth, self.field_size, self.field_size))

			# turn filters into a new volume
			filters.append(volume(filter_volume))
			
		# assign as an instance variable
		self.filters = filters

	# a method that runs the convolution process on a given input volume from the previous layer
	def forward(self, input_volume):
		# check to make sure the input is what the layer expects
		print "HERE:", self.in_height, self.in_width, self.in_depth
		print "HERE:", self.out_height, self.out_width, self.out_depth
		self.check_input_dimensions(input_volume)

		# add padding to input volume if specified (@me: consider not doing a variable reassignment - could be expensive computationally)
		input_volume = self.add_padding(input_volume)

		# go through all the filters, storing the result of a convolution in an output array
		output_slice = []
		for filter_volume in self.filters:
			for i in range(self.out_height):
				# find the current 'y' position of the input image
				row = i * self.stride

				for j in range(self.out_width):
					# find the current 'x' position of the input image
					col = j * self.stride
					# element-wise multiply the depth column by the filters to yield a single value from the convolution
					# find the inputs for a single convolution step
					depth_column = input_volume.volume_slices[:,row:row + self.field_size, col:col + self.field_size]
					output_slice.append(np.sum(depth_column * filter_volume.volume_slices) + 0.1)
		# return the result of a convolution on the entire volume
		return volume(np.reshape(output_slice, (self.out_depth, self.out_height, self.out_width)))
