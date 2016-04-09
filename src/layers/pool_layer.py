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
		# save for backprop
		self.input_volume = input_volume
		
		# store the max's (x,y) position in the original input volume
		# this will later be used to transfer gradients in backprop
		max_row_positions = []
		max_col_positions = []
		
		# output volume of the pooling process
		output_volume = []

		# iterate along the depth dimension of the volume
		for volume_slice in input_volume.volume_slices:
			# iterate along the rows (height dimension)
			for i in range(self.out_height):
				row = i * self.field_size
				# iterate along the columns (width dimension)
				for j in range(self.out_width):
					col = j * self.field_size
					# find the inputs within the current receptive field
					inputs = volume_slice[row:row+self.field_size,col:col+self.field_size]
					
					# find the max's (x,y) position local to the receptive field
					# add the column and row to find max's position within the input
					max_positions = np.unravel_index(inputs.argmax(), inputs.shape)
					max_row_positions.append(max_positions[0] + row)
					max_col_positions.append(max_positions[1] + col)

					# add the max value to the final output volume
					output_volume.append(np.max(inputs))
		
		# reshape arrays into volumes (both saved as instance variables for backprop)
		self.output_volume = volume(np.reshape(output_volume, (self.out_depth, self.out_height, self.out_width)))
		self.max_row_positions = np.reshape(max_row_positions, (self.out_depth, self.out_height, self.out_width))
		self.max_col_positions = np.reshape(max_col_positions, (self.out_depth, self.out_height, self.out_width))

		return self.output_volume

	def backward(self):
		# zero-out existing gradients
		# self.input_volume.zero_gradient()

		input_gradient = np.zeros((self.in_depth, self.in_height, self.in_width))

		# go through each slice of the input and output
		for z in range(self.out_depth):
			# go through the output volume row by column
			for y in range(self.out_height):
				for x in range(self.out_width):
					# referencing instance variables a TON. Maybe change this!
					chain = self.output_volume.gradient_slices[z][y][x]
					max_row = self.max_row_positions[z][y][x]
					max_col = self.max_col_positions[z][y][x]
					input_gradient[z][max_row][max_col] = chain

		self.input_volume.gradient_slices = input_gradient
		return self.input_volume

	def train(self, rate):
		return

	def params_grads(self):
		return []