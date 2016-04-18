import load_cifar
import net_network
import numpy as np

class net_trainer: 
	def __init__(self, network, learning_rate):
		self.network = network
		self.learning_rate = learning_rate
		cifar_data = load_cifar.images_to_volumes("image_data/data_batch_1")
		cifar_test = load_cifar.images_to_volumes("image_data/data_batch_2")

		self.image_volumes = cifar_data[0]
		self.image_labels = cifar_data[1]

		self.test_volumes = cifar_test[0]
		self.test_labels = cifar_test[1]

		self.grad_accumated = []
		self.update_accumated = []

		self.rho_decay = 0.95
		self.epsilon = 0.000001

		self.weight_decay = 0.0001

		train_amount = 1000



		self.train_counter = 1
		for i in range(train_amount):
			print i, "training"
			self.train(self.image_volumes[i], self.image_labels[i])
		network.save_network()
		self.test()

	def train(self, image_volume, image_label):
		self.network.forward(image_volume)
		# loss = self.network.backward(image_label)
		params_grads = self.network.params_grads()
		# print params_grads[1]["grads"]
		self.network.backward(image_label)

		params_grads = self.network.params_grads()

		# print params_grads[2]["grads"]




		if self.train_counter % 2 == 0:
			self.update_params()
		self.train_counter += 1




		# self.network.forward(image_volume)
		# loss = self.network.backward(image_label)
		# params_grads = self.network.params_grads()

		# if len(self.grad_accumated == 0):
		# 	# set the gradient accumations
		# 	for param_grad in params_grads:
		# 		self.grad_accumated.append(np.zeros(param_grad.shape))
		# 		self.update_accumated.append(np.zeros(param_grad.shape))













		# for param_grad in params_grads:



		# 	param = param_grad["params"]
		# 	grad = param_grad["grads"]
			
		# 	# print "Type, ", param_grad["instance"]
		# 	# input_vol = param_grad["instance"].input_volume.volume_slices
		# 	# print "Input (", str(len(input_vol)), str(len(input_vol[0])), str(len(input_vol[0][0])) + " )"
		# 	# print param_grad["instance"].input_volume.volume_slices

		# 	# print "Output"
		# 	# print param_grad["instance"].output_volume.volume_slices, "\n"

		# 	# print "Biases"
		# 	# print param_grad["instance"].biases.volume_slices, "\n"

		# 	# print "Parameter"
		# 	# print param, "\n"
		# 	# print "Gradient"
		# 	# print grad, "\n"
		# 	param += -(self.learning_rate * grad)

		# 	# print "Update"
		# 	# print param, "\n"
		# 	# raw_input("Press Enter to continue...")
		# 	# print param, "after"
		# 	# quit()

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

	def test(self):
		correct = 0
		test_amount = 1000
		for i in range(test_amount):
			print i, "testing"
			result = self.network.classify(self.test_volumes[i])
			print result, self.test_labels[i]
			if result == self.test_labels[i]:
				correct += 1
		print correct / (test_amount * 1.0)*100, "accuracy"
