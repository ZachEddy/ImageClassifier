import load_cifar
import net_network
import numpy as np

class net_trainer: 
	def __init__(self, network, learning_rate):
		self.network = network
		self.learning_rate = learning_rate
		
		# some constants to add as params into the initialize method later
		self.rho_decay = 0.95
		self.epsilon = 0.000001
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

		# define an amount to train with
		train_amount = 2000

		# training over every batch
		self.train_counter = 1
		for i in range(train_amount):
			print i, "training", 1
			self.train_adadelta(self.vol_one[i], self.lab_one[i])

		# for i in range(train_amount):
		# 	print i, "training", 2
		# 	self.train(self.vol_two[i], self.lab_two[i])

		# for i in range(train_amount):
		# 	print i, "training", 3
		# 	self.train(self.vol_three[i], self.lab_three[i])

		# for i in range(train_amount):
		# 	print i, "training", 4
		# 	self.train(self.vol_four[i], self.lab_four[i])

		network.save_network()
		self.test()

	# a function to train using Matt Zeiler's adadelta method - better than standard SGD
	def train_adadelta(self, image_volume, image_label):
		self.network.forward(image_volume)
		self.network.backward(image_label)
		# do batch stuff here if you want
		self.update_params()

	# a helper method for the adadelta training method
	def update_params(self):
		# grad parameters and their gradients
		params_grads = self.network.params_grads()

		# make sure gradient accumulators are initialized
		if len(self.grad_accumated) == 0:
			for param_grad in params_grads:
				param = param_grad["params"]
				self.grad_accumated.append(np.zeros(param.shape))
				self.update_accumated.append(np.zeros(param.shape))

		for i in range(len(params_grads)):
			# get params and their corresponding gradients
			param = params_grads[i]["params"]
			grad = params_grads[i]["grads"] * (self.learning_rate)
			# get existing accumulators for the specific params and gradients
			grad_acc = self.grad_accumated[i]
			update_acc = self.update_accumated[i]


			grad += self.weight_decay * param
			# update gradient accumulator
			grad_acc = grad_acc * self.rho_decay + ((1-self.rho_decay) * (grad * grad))
			# determine update for the parameter
			update = (np.sqrt((update_acc + self.epsilon)/(grad_acc + self.epsilon)) * grad) * -1
			# update the param accumulator
			update_acc = update_acc * self.rho_decay + ((1 - self.rho_decay) * (update * update))
			self.update_accumated[i] = update_acc
			self.grad_accumated[i] = grad_acc
			
			param += update
			params_grads[i]["grads"].fill(0.0)

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

	def test(self):
		correct = 0
		test_amount = 1000
		for i in range(test_amount):
			print i, "testing"
			result = self.network.classify(self.vol_test[i])
			print result, self.lab_test[i]
			if result == self.lab_test[i]:
				correct += 1
		print correct / (test_amount * 1.0)*100, "accuracy"
