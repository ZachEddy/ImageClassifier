# a class to manage the pooling layer
from src.net_utilities import is_int

class pool_layer:
	def __init__(self, field_size, in_height, in_width, in_depth):
		# set params as instance variables
		self.field_size = field_size
		self.in_height = in_height
		self.in_width = in_width
		self.in_depth = in_depth
		self.calc_output_dimensions()

	def calc_output_dimensions(self):
		# define the output volume dimensions and make sure the inputs are valid
		self.out_height = self.in_height / (self.field_size * 1.0)
		self.out_width = self.in_height / (self.field_size * 1.0)
		self.out_depth = self.in_depth

		# ensure integer dimensions for output volume
		if not(is_int(self.out_height)) or not(is_int(self.out_width)):
			print "Input dimensions into pool layer aren't valid"
			exit()

	def forward(self, input_volume):
		# I don't do anything yet
		return

	def backward(self, input_volume):
		# I don't do anything yet
		return