<h1>BacteriaNet</h1>
<b>A neural network implemented in Python to classify images of microorganisms</b>
<br>
I'm building a convolutional neural network to identify images of single-celled organisms. To get things rolling, I've started by using the CIFAR-10 dataset. This contains five batches of 10,000 32x32 pixel images. Each of the 10 training categories (cars, frogs, boats, etc.) has 1,000 images each.
<br>

Once I get it working, I'd like to mount it on a Django app that lets users feed images through the ConvNet themselves. I don't have any attachment specifically to Django, but it'll keep everything homogenously in Python.
<br>

We'll see how it goes!
<h4> Reference </h4>
<ul>
  <li>
   <a href="https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf">Learning Multiple Layers of Features from Tiny Images</a>, Alex Krizhevsky, 2009.
  </li>
</ul>
