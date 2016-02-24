# a class to manage the pooling layer
from src.net_utilities import is_int
import numpy as np
from src.volume import volume

class pool_layer:
	def __init__(self, field_size, in_height, in_width, in_depth):
		# set params as instance variables
		self.field_size = field_size
		self.in_height = in_height
		self.in_width = in_width
		self.in_depth = in_depth
		self.calc_output_dimensions()

	# a function to make sure the inputs will produce a valid output
	def check_user_definition(self):
		if not(is_int(self.out_height)) or not(is_int(self.out_width)):
			print "Input dimensions into pool layer aren't valid"
			quit()

	# a function to calculate the output dimensions of a volume after pooling has been done
	def calc_output_dimensions(self):
		# define the output volume dimensions and make sure the inputs are valid
		self.out_height = self.in_height / (self.field_size * 1.0)
		self.out_width = self.in_height / (self.field_size * 1.0)
		self.out_depth = self.in_depth

		# ensure integer dimensions for output volume
		self.check_user_definition()

		# assuming the output volumes are integer values, then make them integers in computer terms (ex: 6.0 --> 6) 
		self.out_height = int(self.out_height)
		self.out_width = int(self.out_width)
		self.out_depth = int(self.out_depth)

	# a function that feeds a given volume through the pooling layer
	def forward(self, input_volume):	
		output_volume = []
		# iterate along the depth dimension of the volume
		for volume_slice in input_volume.volume_slices:
			# iterate along the rows (height dimension)
			for i in range(self.out_height):
				row = i * self.field_size
				# iterate along the columns (width dimension)
				for j in range(self.out_width):
					col = j * self.field_size
					# max pool the inputs within the receptive field
					inputs = volume_slice[row:row+self.field_size,col:col+self.field_size]
					output_volume.append(np.max(inputs))

		# return the new volume reshaped with the expected output dimensions
		return volume(np.reshape(output_volume, (self.out_depth, self.out_height, self.out_width)))

	def backward(self, input_volume):
		# I don't do anything yet
		return