# a class to manage the convolution layer
import numpy as np
from src.volume import volume
from src.net_utilities import is_int
	
class conv_layer:
	def __init__(self, name, field_size, filter_count, stride, padding, in_height, in_width, in_depth):
		# set name for weight saving/loading purposes
		self.name = name

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

	# a function to add a padded border around the edges of a volume along the width and height dimensions
	def add_padding(self, vol):
		# minimize number of references to instance variable 'self.padding'
		padding = self.padding
		if padding > 0:
			pad_specs = ((0,0), (padding, padding), (padding, padding))
			vol = np.pad(vol, pad_specs, mode='constant',constant_values=0)
		return vol

	# a function to remove padding border around a volume
	def trim_padding(self, vol):
		padding = self.padding
		if padding > 0:
			new_vol = vol[:,padding:len(vol[0])-padding,padding:len(vol[0][0])-padding]
			return new_vol
		return vol
	
	# a function to calculate output volume height based on input volume height
	def calc_output_height(self):
		return (self.in_height - self.field_size + 2.0 * self.padding) / self.stride + 1.0

	# a function to calculate output volume width based on input volume width
	def calc_output_width(self):
		return (self.in_width - self.field_size + 2.0 * self.padding) / self.stride + 1.0

	# a function to check user-defined layer specs will produce a valid output (no non-integer pixel values)
	def check_user_definition(self):
		# quit the program if any of the dimensions aren't integers
		# print self.out_height, self.out_width, self.out_depth
		if not(is_int(self.out_height)) or (not is_int(self.out_width)) or (not is_int(self.out_depth)):
			print "~~ Input dimensions into conv layer aren't valid"
			quit()

	# a function to make sure that inputs into the conv layer match with what it expects
	def check_input_dimensions(self, input_volume):
		if (input_volume.height != self.in_height) or (input_volume.width != self.in_width) or (input_volume.depth != self.in_depth):
			print "~~ Dimensions of input volume do not match expected dimensions"
			quit()

	# a function to calculate the dimensions of the output volume
	def calc_output_dimensions(self):
		# define the output volume dimensions and make sure the inputs are valid
		self.out_height = self.calc_output_height()
		self.out_width = self.calc_output_width()
		self.out_depth = self.filter_count

		# check to make sure the user enterered valid layer specs (no part-pixels)
		self.check_user_definition()
		
		# assuming the output volumes are integer values, then make them integers in computer terms (ex: 6.0 --> 6) 
		self.out_height = int(self.out_height)
		self.out_width = int(self.out_width)
		self.out_depth = int(self.out_depth)

	def initialize_biases(self):
		# make a matrix with the dimensions of the output volume
		biases = np.zeros((1,1,self.out_depth))

		# fill the matrix with 0.1 (initial bias value) and make it an instance variable
		biases.fill(0.1)
		self.biases = volume(biases)

	# a function to initialize the filter volumes
	def initialize_filters(self):
		# a list to hold all the filter volumes - depth of volumes is the depth of the input volume
		# a filter will be paired with inputs from each layer of depth
		filters = []

		# calculate the area of the receptive field
		weight_count = (self.field_size ** 2) * self.in_depth

		# create filter volumes for the number of sets the user specifies. If depth is 3, and the user specifies
		# 20 filter sets, then there will be 60 total filter. If the receptive field size is 3, then the 
		# total number of weights will be 540. (3 * 20 * (3^2))
		for i in range(self.filter_count):
			# initialize weights for every filter
			# recommended initalization function for ReLu: np.random.randn(n) * np.sqrt(2.0 / n)
			filter_volume = np.random.randn(weight_count) * np.sqrt(2.0 / weight_count)
			filter_volume = np.reshape(filter_volume, (self.in_depth, self.field_size, self.field_size))

			# turn filters into a new volume
			filters.append(volume(filter_volume))
			
		# assign as an instance variable
		self.filters = filters

	# a function to initialize the network biases
	def initialize_params(self):
		self.initialize_biases()
		self.initialize_filters()

	# a function to save weights and biases
	def save_params(self,net_name):
		filter_params = []
		for filter_volume in self.filters:
			filter_params.append(filter_volume.volume_slices)
		np.savez("saved_networks/"+net_name+"/"+self.name, biases = self.biases.volume_slices, filters = filter_params)

	# a function to load existing weights from a file
	def load_params(self, net_name):
		params = np.load("saved_networks/"+net_name+"/"+self.name + ".npz")
		filters = []
		for filter_values in params["filters"]:
			filters.append(volume(filter_values))
		self.filters = filters
		self.biases = volume(params["biases"])

	# a function that runs the convolution process on a given input volume from the previous layer
	def forward(self, input_volume):
		# check to make sure the input is what the layer expects
		self.check_input_dimensions(input_volume)
		
		# make instance input_volume an unmodified version of original input
		self.input_volume = volume(input_volume.volume_slices)

		# add padding to input volume if specified
		input_volume.volume_slices = self.add_padding(input_volume.volume_slices)

		# go through all the filters, storing the result of a convolution in an output array
		output_slice = []
		stride = self.stride
		for filter_volume, bias in zip(self.filters, self.biases.volume_slices[0][0]):
			for i in range(self.out_height):
				# find the current 'y' position of the input image
				row = i * self.stride
				for j in range(self.out_width):
					# find the current 'x' position of the input image
					col = j * self.stride
					# element-wise multiply the depth column by the filters to yield a single value from the convolution
					# find the inputs for a single convolution step
					depth_column = input_volume.volume_slices[:,row:row + self.field_size, col:col + self.field_size]
					output_slice.append(np.sum(depth_column * filter_volume.volume_slices) + bias)

		# return the result of a convolution on the entire volume
		self.output_volume = volume(np.reshape(output_slice, (self.out_depth, self.out_height, self.out_width)))
		return self.output_volume

	# backpropagation for convolution layer
	def backward(self):
		# create a padded matrix with zeros to hold input gradients
		input_padding = self.padding * 2
		input_gradient = np.zeros((self.in_depth, self.in_height + input_padding, self.in_width + input_padding))

		# do the same for the biases
		biases_gradient = np.zeros((1,1,self.out_depth))

		# preserve the original input volume before making changes
		input_volume = volume(self.input_volume.volume_slices)
		input_volume.volume_slices = self.add_padding(input_volume.volume_slices)

		# iterate through the entire volume (height, width, depth); first along the depth dimension
		# this also functions to iterate through each filter individually
		for depth_index in range(self.out_depth):
			# iterate along the height dimension
			# create an empty volume of zeros for the gradient
			filter_gradient = np.zeros((self.in_depth, self.field_size, self.field_size))

			for height_index in range(self.out_height):
				# iterate along the width dimension
				# make sure to find row position in input volume
				row = height_index * self.stride
				
				for width_index in range(self.out_width):
					# iterate along the height dimension
					# make sure to find column position in input volume
					col = width_index * self.stride

					# find the gradient from the previous layer
					chain_gradient = self.output_volume.gradient_slices[depth_index][height_index][width_index]
					
					# update filter gradient by multiplying previous layer's gradient 
					# with the inputs within the receptive field
					depth_column = input_volume.volume_slices[:,row:row + self.field_size, col:col + self.field_size]
					filter_gradient += depth_column * chain_gradient
					
					# add the filter weights multiplied by the chain to the input
					input_gradient[:,row:row + self.field_size, col:col + self.field_size] += self.filters[depth_index].volume_slices * chain_gradient

					# update the biases
					biases_gradient[0][0][depth_index] += chain_gradient

			# set the new gradients to whatever filter we're currently iterating over
			# chnaged to +=
			self.filters[depth_index].gradient_slices += filter_gradient

		# trim the padding off the gradient volume to match the input volume's original size
		input_gradient = self.trim_padding(input_gradient)
		
		# set the input and bias gradients to whatever gradients we just calculated
		self.input_volume.gradient_slices = input_gradient
		# changed to += 
		self.biases.gradient_slices += biases_gradient
		return self.input_volume

	# a function to return params (weights and biases) and their gradients for training
	def params_grads(self):
		aggregate = []
		aggregate.append({"params":self.biases.volume_slices, "grads":self.biases.gradient_slices, "instance":self})
		for i in range(len(self.filters)):
			aggregate.append({"params":self.filters[i].volume_slices, "grads":self.filters[i].gradient_slices, "instance":self})
		return aggregate
