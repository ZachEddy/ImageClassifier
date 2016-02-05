# input layer should be in charge of modifying pixel values (0-255)
# this will include normalization and mean value subtraction
class input_layer:
	def __init__(self, out_height, out_width, out_depth):
		self.out_height = out_height
		self.out_width = out_width
		self.out_depth = out_depth

	def normalize_layer(self):
		return

# function of the input layer:
	# take in an image volume, normalize it, and pass it on.
	# it should have a width, height, and depth dimension
	# defined by the user in the net_initialize module