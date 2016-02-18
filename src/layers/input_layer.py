import numpy as np
# input layer should be in charge of modifying pixel values (0-255)
# this will include normalization and mean value subtraction
class input_layer:
	def __init__(self, out_height, out_width, out_depth):
		# define output dimensions
		self.out_height = out_height
		self.out_width = out_width
		self.out_depth = out_depth

	
	def forward(self, input_volume):
		# this takes an input and normalizes the pixel values (0-255) to be something from -0.5 to 0.5
		for volume_slice in input_volume.volume_slices:
			volume_slice = self.normalize_matrix(volume_slice)

	def normalize_pixel(pixel_value):
		# normalize pixel values between 0.5 and -0.5
		return pixel_value/255.0 - 0.5

	# vectorizing the normalize_pixel function allows me to easy map it over an entire matrix
	# rather than using a nested for loop (it's the same process under the hood, though)
	normalize_matrix = np.vectorize(normalize_pixel)