<h1>ZachNet</h1>
A Convolutional Neural Network implemented in Python to classify images in the following categories:
  - Airplanes
  - Automobiles
  - Birds
  - Cats
  - Deer
  - Dogs
  - Frogs
  - Horses
  - Ships
  - Trucks

<h2>Prerequisites</h2>
**Numpy** - a powerful math library written in Python. If you haven't already installed it, running this command *should* bring your machine up to speed:

`pip install numpy`

More information about installing numpy available [here](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html).

<h2>Installing</h2>
Just clone this repository, and you're set to go! **Quick note**: this includes the entire CIFAR-10, a compressed dataset of 50,000 32x32 pixel images. If the cloning process takes a few minutes, that's why. 


`git clone https://github.com/ZachEddy/ImageClassifier`

<h2>Running the code</h2>
<h4>Quickstart </h4>



Feel free to run `python __main__.py` at the top-level directory to get things going. I have the network defaulted to load a pretrained network and run on ten testing images.

<h4>Create your own network</h4>

I made it easy to train, save, and load your own network. Inside `net_initialize.py`, you can create networks with different layer patterns. Each layer has a few user-defined parameters. For example, you can change the number of filters in a convolution layer with the following:

`{'type':'conv', 'field_size':5, 'filter_count':10, 'stride':1, 'padding':2, 'name':"conv_one"}`

`{'type':'conv', 'field_size':5, 'filter_count':15, 'stride':1, 'padding':2, 'name':"conv_one"}`

Changing the layer pattern and/or their associated layer parameters will impact:

1. **network accuracy** - how consistently the network classifies correctly.
2. **training time** - how long it takes the network to learn.

Lastly, this line of code inside `net_initialize` creates a new network based on the *layer structure* provided by the user:

`return net_network(layer_structure,"network_name")`

After training completes, the weights, biases, and layer structure get saved as `network_name` in the `saved_networks` directory.

<h4>Load a network</h4>

Alternatively, you can load a network to classify something *after* the training process finishes. 

`return net_network(None,"pretrained_network_name")`

In this case, `None` informs the network that no layer structure has been provided by the user. It will search for `pretrained_network` inside the `saved_networks` directory instead. 

<hr>

Take a look at the `net_initialize.py` code; I think this will make it much more conceptually concrete. Also, having a basic understanding of Convolutional Neural Networks will really help. I recommend [this detailed explanation](http://cs231n.github.io/convolutional-networks/) from a Stanford course on ConvNets.







<h2> Reference </h2>
<ul>
  <li>
   <a href="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf">Learning Multiple Layers of Features from Tiny Images</a>, Alex Krizhevsky, 2009.
  </li>
</ul>
