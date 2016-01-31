import numpy as np
class volume:
	
	def __init__(self, width, height):
		self.width = width
		self.height = height
		self.volume_slices = []

	def add_volume_slice(self, *args):
		for volume_slice in enumerate(args):
			self.volume_slices.append(volume_slice)