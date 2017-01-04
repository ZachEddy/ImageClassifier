<h1>Image Classifier</h1>
A convolutional neural network implemented in Python to classify images in the following ten categories:
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

<h2>Background</h2>
Neural networks take input(s), perform computation in sequential layers, then generate output(s). For example, a network could determine an athlete's sport given height, weight, and favorite Gatorade flavor. It may find that most heavy-set tall people play football, and most lightweight tall people run long-distance. Gatorade flavor presumably won't influence classification. For instance, it seems unlikely that most football players prefer grape, while runners prefer strawberry. Physical attributes offer better insight in this context, not matters of personal preference &mdash; this is something a neural network can <i>learn</i>. However, the network has to undergo training before it can make classifications accurately. This process involves individually inputting hundreds, thousands, or potentially millions of examples with known outcomes. Each time the network classifies incorrectly, it will adjust slightly with the goal of improving accuracy.

Convolutional networks are similar, but function for three-dimensional inputs. Images are two-dimensional along height and width. They form a third, depth-wise dimension with RGB color channels. This particular network can examine images, find recurrent patterns, then classify based on what it learned. For example, after looking at hundreds of horses, it will eventually discover the common features &mdash; ears, noses, hoofs, etc.

<b>TLDR:</b> Neural networks make informed classifications between input and ouput after training on example data. Basic neural networks are one-dimensional, convolutional neural networks are three-dimensional and more complex.

<h2>Dependencies</h2>
**NumPy** - a math library written in Python. You can install it via command-line if necessary:

    $ pip install numpy

More information about setting up NumPy [here](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html).

<h2>Installing</h2>
Clone the repository and you're set.

    $ git clone https://github.com/ZachEddy/ImageClassifier

**Note**: this includes the CIFAR-10, a compressed set of 50,000 32x32 images. The cloning could take 3-5 minutes.

<h2>Running</h2>
<h4>Quickstart</h4>

Run `python __main__.py` at the top-level directory. I have the network defaulted to load a pre-trained network and classify ten images from a testing set. The default network trained overnight with 40,000 total images.

<h4>Create your own network</h4>

You can easily train, save, and load your own network. Inside `net_initialize.py`, you can create networks with different layer patterns. Each layer has a few user-defined parameters. For example, you can change the number of filters in a convolution layer with the following:

    {'type':'conv', 'field_size':5, 'filter_count':10, 'stride':1, 'padding':2, 'name':"conv_one"}
    {'type':'conv', 'field_size':5, 'filter_count':15, 'stride':1, 'padding':2, 'name':"conv_one"}

Changing the layer pattern and/or their associated layer parameters will impact:

1. **network accuracy** - how consistently the network classifies correctly.
2. **training time** - how long it takes the network to learn.

Lastly, this line of code inside `net_initialize` creates a new network based on the *layer structure* provided by the user:

    return net_network(layer_structure,"network_name")

After training completes, the weights, biases, and layer structure get saved as `network_name` in the `saved_networks` directory.

<h4>Load a network</h4>

Alternatively, you can load a network to classify something *after* the training process finishes.

    return net_network(None,"pretrained_network_name")

`None` informs the network that no layer structure has been provided by the user. It will search for `pretrained_network` inside the `saved_networks` directory instead.

<hr>

Take a look at the `net_initialize.py` code; I think this will make it much more conceptually concrete. Also, having a basic understanding of Convolutional Neural Networks will really help. I recommend [this detailed explanation](http://cs231n.github.io/convolutional-networks/) from a Stanford course on ConvNets.







<h2> Reference </h2>
<ul>
  <li>
   <a href="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf">Learning Multiple Layers of Features from Tiny Images</a>, Alex Krizhevsky, 2009.
  </li>
</ul>
