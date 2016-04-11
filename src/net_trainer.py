import load_cifar
import net_network

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

		train_amount = 1000

		# for i in range(train_amount):
		# 	print i, "training"
		# 	self.train(self.image_volumes[i], self.image_labels[i])
		# network.save_network()
		self.test()

	def train(self, image_volume, image_label):
		self.network.forward(image_volume)
		loss = self.network.backward(image_label)
		params_grads = self.network.params_grads()
		# print loss
		# print loss, image_label, self.network.layers[len(self.network.layers)-1].classify
		counter = 0
		for param_grad in params_grads:
			param = param_grad["params"]
			grad = param_grad["grads"]
			
			# print "Type, ", param_grad["instance"]
			# input_vol = param_grad["instance"].input_volume.volume_slices
			# print "Input (", str(len(input_vol)), str(len(input_vol[0])), str(len(input_vol[0][0])) + " )"
			# print param_grad["instance"].input_volume.volume_slices

			# print "Output"
			# print param_grad["instance"].output_volume.volume_slices, "\n"

			# print "Biases"
			# print param_grad["instance"].biases.volume_slices, "\n"

			# print "Parameter"
			# print param, "\n"
			# print "Gradient"
			# print grad, "\n"
			param += -(self.learning_rate * grad)

			# print "Update"
			# print param, "\n"
			# raw_input("Press Enter to continue...")
			# print param, "after"
			# quit()


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
