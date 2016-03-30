# a class to hold volumes of numerical values
# this could also be described as a 3D array 
import numpy as np
class volume:	
	def __init__(self, volume):
		# initialize height, width, and depth according to the 3D array entered by the user
		self.width = len(volume[0][0])
		self.height = len(volume[0])
		self.depth = len(volume)
		self.volume_slices = volume
		self.zero_gradient()

	# a function to reset the gradient after each backprop cycle
	def zero_gradient(self):
		self.gradient_slices = np.zeros((self.depth, self.height, self.width))