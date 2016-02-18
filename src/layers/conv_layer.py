# a class to manage the convolution layer
import numpy as np
from src.volume import volume
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

		# initialize filter volumes
		self.initialize_filters()

	def calc_output_dimensions(self):
		# define the output volume dimensions and make sure the inputs are valid
		self.out_height = (self.in_height - self.field_size + 2.0 * self.padding) / self.stride + 1
		self.out_width = (self.in_width - self.field_size + 2.0 * self.padding) / self.stride + 1
		self.out_depth = self.filter_count

		# ensure integer dimensions for output volume
		if not(is_int(self.out_height)) or (not is_int(self.out_width)) or (not is_int(self.out_depth)):
			print "Input dimensions into conv layer aren't valid"
			exit()

		self.out_height = int(self.out_height)
		self.out_width = int(self.out_width)
		self.out_depth = int(self.out_depth)


	def initialize_filters(self):
		# filters will also be represented as volumes, where each slice corresponds to a slice of the input volume
		# the total number of filter volumes will be the filter_count
		filters = []

		# create as many filter volumes as specified by the user (would be handy to have a 'int.times' method like Ruby, but whatever)
		weight_count = self.field_size ** 2
		for i in range(self.filter_count):

			filter_volume = np.array([])

			
			# 
			for j in range(self.in_depth):
				weights = np.random.randn(weight_count) * np.sqrt(2.0/weight_count)
				filter_volume = np.append(filter_volume, weights)
			
			filter_volume = np.reshape(filter_volume, (self.in_depth, self.field_size, self.field_size))
			filters.append(volume(filter_volume))

		self.filters = filters


	def forward(self, input_volume):
		# I don't do anything yet
		return





	# def field_inputs(column, row):

		# return


	# def 


	def forward(self, volume):
		output_volume = []

		# go through all the filters
		for filter_volume in self.filters:
			


			output_slice = np.empty((self.out_height, self.out_width))


			# generate a slice in the output volume, row by column
			# start by going through all the rows
			print self.out_height
			print self.out_width
			for i in range(self.out_height):
				row = i * self.stride
				
				# for all the rows, go through all the columns
				for j in range(self.out_width):
					col = j * self.stride

					depth_column = volume[:,row + self.stride, col + self.stride]
					conv_calculation = np.sum(depth_column * filter_volume.volume_slices)
					output_slice = np.append(output_slice, conv_calculation)

			output_slice = output_slice.reshape(self.out_height, self.out_width)
			output_volume.append(output_slice)
			quit()

		output_volume = np.array(output_volume)
		return output_volume
















		# # traverse each row of the matrix (rows correspond to the height dimension)
		# for i in range(out_height):
		# 	row = i * stride
		# 	# traverse each column of the matrix (rows correspond to the height dimension)
		# 	for j in range(out_width):
		# 		# inputs into the filters
		# 		input_neurons = []
		# 		col = j * stride

		# 		# k to the rows within the receptive field
		# 		for k in field_size:
		# 			input_neurons.append(volume_slice[i + k][j:j+field_size])

		# return

	def backward(self, input_volume):
		# I don't do anything yet
		return