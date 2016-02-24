# define the organization of layers in the network
# to make it flexible to changes in structure
from src.net_network import net_network

# ordered structure of layers in the neural network
layer_structure = []

# continue to add layers (ReLu, conv, pool, softmax, etc)
layer_structure.append({'type':'input', 'out_height':32, 'out_width':32, 'out_depth':3})

layer_structure.append({'type':'conv', 'field_size':5, 'filter_count':16, 'stride':1, 'padding':2})
layer_structure.append({'type':'relu'})

layer_structure.append({'type':'pool', 'field_size':2})

layer_structure.append({'type':'conv', 'field_size':5, 'filter_count':20, 'stride':1, 'padding':2})
layer_structure.append({'type':'relu'})

layer_structure.append({'type':'pool', 'field_size':2})

layer_structure.append({'type':'conv', 'field_size':5, 'filter_count':20, 'stride':1, 'padding':2})
layer_structure.append({'type':'relu'})

layer_structure.append({'type':'pool', 'field_size':2})

layer_structure.append({'type':'fully_connected', 'neuron_count':100})
layer_structure.append({'type':'relu'})

layer_structure.append({'type':'fully_connected', 'neuron_count':10})

layer_structure.append({'type':'tanh'})

# create a network with the layer specificications
net = net_network(layer_structure)