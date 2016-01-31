# define the organization of layers in the network
# to make it flexible to changes in structure
from src.net_network import net_network

# ordered structure of layers in the neural network
layer_structure = []

# continue to add layers (ReLu, conv, pool, softmax, etc)
layer_structure.append({'type':'conv', 'field_size':3, 'filter_count':16, 'stride':1, 'padding':2})
layer_structure.append({'type':'pool', 'field_size':2})

# create a network with the layer specificications
j = net_network(layer_structure)