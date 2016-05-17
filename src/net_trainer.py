import load_cifar
import net_network
import numpy as np

class net_trainer: 
	def __init__(self, network, learning_rate):
		self.network = network
		self.learning_rate = learning_rate
		self.weight_decay = 0.0001

		# accumulators for the gradient and parameters (used in adadelta)
		self.grad_accumated = []
		self.update_accumated = []

		# import all the batches
		batch_one = load_cifar.images_to_volumes("image_data/data_batch_1")
		batch_two = load_cifar.images_to_volumes("image_data/data_batch_2")
		batch_three = load_cifar.images_to_volumes("image_data/data_batch_3")
		batch_four = load_cifar.images_to_volumes("image_data/data_batch_4")
		batch_test = load_cifar.images_to_volumes("image_data/data_batch_5")

    # define all the volumes and their labels
		self.vol_one = batch_one[0]
		self.lab_one = batch_one[1]

		self.vol_two = batch_two[0]
		self.lab_two = batch_two[1]

		self.vol_three = batch_three[0]
		self.lab_three = batch_three[1]

		self.vol_four = batch_four[0]
		self.lab_four = batch_four[1]

		self.vol_test = batch_test[0]
		self.lab_test = batch_test[1]

	# a function to run through the testing sets and train the network
	def train(self, train_amount):
		print "~~ Training network with", train_amount, "images from 4 training sets (", (train_amount * 4), "total )"
		raw_input("~~ Press enter to continue...")
		print
		for i in range(train_amount):
			print "~~~~ Training set 1 - progress:",i+1, "/", train_amount
			self.train_sgd(self.vol_one[i], self.lab_one[i])

		for i in range(train_amount):
			print "~~~~ Training set 2 - progress:",i+1, "/", train_amount
			self.train_sgd(self.vol_two[i], self.lab_two[i])

		for i in range(train_amount):
			print "~~~~ Training set 3 - progress:",i+1, "/", train_amount
			self.train_sgd(self.vol_three[i], self.lab_three[i])

		for i in range(train_amount):
			print "~~~~ Training set 4 - progress:",i+1, "/", train_amount
			self.train_sgd(self.vol_four[i], self.lab_four[i])
		print


	# a function to train with SGD alone
	def train_sgd(self, image_volume, image_label):
		# forward and backward propagate the network
		self.network.forward(image_volume)
		loss = self.network.backward(image_label)

		# get the parameters and gradients for everything in the network
		params_grads = self.network.params_grads()

		# go through every layer in the network that has parameters
		for i in range(len(params_grads)):
			# get parameters and gradients at each layer in the network
			param = params_grads[i]["params"]
			grad = params_grads[i]["grads"]
			grad += self.weight_decay * param
			# update the parameters by the gradient, then reset the gradient to zero
			param -= self.learning_rate * grad
			params_grads[i]["grads"].fill(0.0)

	# a method to test the accuracy of the network
	def test(self, test_amount):
		print "~~ Testing accuracy with", test_amount, "images from testing set..."
		raw_input("~~ Press enter to continue...")
		print

		# keep track of how many the network got correct
		correct = 0
		for i in range(test_amount):
			# iterate through the testing set
			print "~~~~ Testing progress...", i+1, "/", test_amount
			result = self.network.classify(self.vol_test[i])
			if result == self.lab_test[i]:
				correct += 1
		print
		print "~~ Testing result:", correct / (test_amount * 1.0)*100, "percent accuracy"
