from neural_network import NN
from mnist import MNIST
import numpy as np
from time import time

def xor_net():
	a = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
	# b = np.tile(a, (2500,1))
	np.random.shuffle(a)
	features = a[:, 0:2].T
	labels = a[:, 2].reshape((1, 4))
	nn = NN([2, 4, 3, 1])
	nn.set_hyperparameters(batch_size=4, learning_rate=0.75)
	nn.initialize_parameters()
	nn.minimize({"features": features, "labels": labels}, 10000)
	print(nn.cost)
	print(nn.evaluate({"features": features, "labels": labels}))


def mnist_net():
	mndata = MNIST("mnist", return_type="numpy")
	print("Loading images...")
	images, labels = mndata.load_training()
	t = time()
	features = images.T/255
	z = np.zeros((60000, 10))
	z[np.arange(60000), labels] = 1
	Y = z.T
	nn = NN([784, 100 , 30, 10])
	nn.set_hyperparameters(learning_rate=0.5)
	nn.initialize_parameters()
	print("Start Training...")
	nn.minimize({"features": features, "labels": Y}, 20)
	print("Finish Training.")
	print("Start Testing...")
	print("Training time: {0} seconds".format(round(time()-t,2)))
	t = time()
	test_images, test_labels = mndata.load_testing()
	test_features = test_images.T/255
	z = np.zeros((10000, 10))
	z[np.arange(10000), test_labels] = 1
	test_Y = z.T
	print("Testing accuracy: {}".format(round(nn.evaluate({"features": test_features, "labels": test_Y}),4)))
	print("Testing time: {0} seconds".format(round(time()-t,2)))
	
if __name__ == "__main__":
	mnist_net()