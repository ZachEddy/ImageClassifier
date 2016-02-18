# a class to hold volumes of numerical values
# this could also be described as a 3D array 

import numpy as np
class volume:	
	# def __init__(self, width, height):
	# 	self.width = width
	# 	self.height = height
	# 	self.volume_slices = []

	def __init__(self, volume):
		self.width = len(volume[0][0])
		self.height = len(volume[0])
		self.volume_slices = volume

	def add_volume_slice(self, *args):
		for volume_slice in enumerate(args):
			# access [1] element because enumerate() puts *args into a tuple, where the other element is the index.
			# yeah... took me forever to figure that one out.
			self.volume_slices.append(volume_slice[1])