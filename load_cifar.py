# a class that turns the cifar-10 images into volumes
import cPickle
import numpy as np
from src.volume import volume

file_path = "image_data/data_batch_1"

# de-serialize the data batch
def unpickle_batch(file_path):
	fo = open(file_path, 'rb')
	batch_data = cPickle.load(fo)
	fo.close()
	return batch_data

# convert cifar-10 image arrays to volumes
def images_to_volumes():
	volumes = []
	batch_data = unpickle_batch(file_path)
	
	# cifar-10 stores 32x32 (1024 total) pixel images as an array of length 3072
	# first 1024 are red, next 1024 are green, final 1024 are blue. Pretty awesome!
	# each array of 3072 is put into an array of 10,000 (because 10,000 total images)
	for rgb_array in batch_data['data']:
		# add each rgb channel as its own layer (depth_slice) to the volume
		r_slice = np.reshape(rgb_array[0:1024], (32,32))
		g_slice = np.reshape(rgb_array[1024:2048], (32,32))
		b_slice = np.reshape(rgb_array[2048:3072], (32,32))
		image_volume = volume(32,32)
		image_volume.add_volume_slice(r_slice, g_slice, b_slice)
		# add to list containing all image volumes
		volumes.append(image_volume)
	return volumes

