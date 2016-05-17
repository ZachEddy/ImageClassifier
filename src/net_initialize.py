# define the organization of layers in the network
# to make it flexible to changes in structure
from src.net_network import net_network

# ordered structure of layers in the neural network
layer_structure = []

# continue to add layers (ReLu, conv, pool, softmax, etc)
layer_structure.append({'type':'input', 'out_height':32, 'out_width':32, 'out_depth':3})

layer_structure.append({'type':'conv', 'field_size':5, 'filter_count':10, 'stride':1, 'padding':2, 'name':"conv_one"})
layer_structure.append({'type':'relu'})

layer_structure.append({'type':'pool', 'field_size':2})

layer_structure.append({'type':'conv', 'field_size':5, 'filter_count':10, 'stride':1, 'padding':2, 'name':"conv_two"})
layer_structure.append({'type':'relu'})

layer_structure.append({'type':'pool', 'field_size':2})

layer_structure.append({'type':'conv', 'field_size':5, 'filter_count':10, 'stride':1, 'padding':2, 'name':"conv_three"})
layer_structure.append({'type':'relu'})

layer_structure.append({'type':'pool', 'field_size':2})

layer_structure.append({'type':'fully_connected', 'neuron_count':100, 'name':"fc_one"})
layer_structure.append({'type':'relu'})

layer_structure.append({'type':'fully_connected', 'neuron_count':10, 'name':"fc_two"})

layer_structure.append({'type':'softmax'})

# generate a new neural network with the given specs
def build():
	# generate untrained network; give the network a name
	# return net_network(layer_structure,"new_network_name")

	# alternatively, load an existing network with 'None' as the layer specs
	# and 'saved_network_name' as the name of the network.
	# 'sgd_one' is the name of the trained network that I used for the demo (50.4% accuracy on
	# 1000 images from the testing set)
	
	return net_network(None,"sgd_one")