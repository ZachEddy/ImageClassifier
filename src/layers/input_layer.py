import numpy as np
from src.volume import volume
# input layer should be in charge of modifying pixel values (0-255)
# this will include normalization and mean value subtraction
class input_layer:
	def __init__(self, out_height, out_width, out_depth):
		# define output dimensions
		self.out_height = out_height
		self.out_width = out_width
		self.out_depth = out_depth
		# vectorize the pixel normalization function so it can easily be mapped over matrices
		self.normalize_volume = np.vectorize(self.normalize_pixel)
 
	# a function to noramlize a single pixel values
	def normalize_pixel(self, pixel_value):
		# normalize pixel values between 0.5 and -0.5
		return pixel_value/255.0 - 0.5

	# the feed-forward function for an input layer - it's extremely simple
	def forward(self, input_volume):
		return volume(self.normalize_volume(input_volume.volume_slices))