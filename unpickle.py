import os
import sys
import numpy as np
sys.path.append(os.getcwd())
import volume as volume



def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def image_volumes(dict):
	volumes = []





for x in xrange(1,10):
	pass




dictionary = unpickle("data_batch_2")
# print dictionary
images = dictionary['data']
# print images
image1 = images[0]


# print images, "here"
# print image1['data'][0]



red = np.array(image1[0:1024])
# print red[0]
# print red
red = np.reshape(red,(32,32))
print red
print red[0][31]
print len(red[0])