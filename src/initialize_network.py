from layers.conv_layer import conv_layer
from image_data import test


layer_specs = []

typf = conv_layer(4,4,5,5)

# continue to add layers (ReLu, Conv, pool, softmax)
layer_specs.append({'type':'conv', 'field_size':3, 'filter_count':16, 'stride':1, 'pad':2, 'activation':'relu'})
layer_specs.append({'type':'pool', 'field_size':2, 'stride':2})




print layer_specs[0]