import numpy as np


class NN:
	def __init__(self, layers):
		self.learning_rate = 0.001
		self.batch_size = 64
		self.layers = layers
		self.weights = {}
		self.biases = {}
		self.output = np.array([])
		self.train_features = np.array([])
		self.train_Y = np.array([])
		self.test_features = np.array([])
		self.test_Y = np.array([])
		self.cache = {}
		self.grads = {}
		self.cost = np.array([])
		self.accuracy = 0
	
	def initialize_parameters(self):
		for i in range(1, len(self.layers)):
			self.weights["W" + str(i)] = np.random.randn(self.layers[i], self.layers[i - 1])
			self.biases["b" + str(i)] = np.zeros((self.layers[i], 1))
	
	def feed_train_input(self, features, labels):
		self.train_features = features
		self.train_Y = labels
	
	def feed_test_features(self, features):
		self.test_features = features
	
	def feed_test_labels(self, labels):
		self.test_Y = labels
	
	def set_hyperparameters(self, learning_rate=0.001, batch_size=64):
		self.learning_rate = learning_rate
		self.batch_size = batch_size
	
	def forward_prop(self, X):
		A = X
		i = 0
		for i in range(1, len(self.layers)):
			self.cache["A" + str(i - 1)] = A
			Z = np.matmul(self.weights["W" + str(i)], A) + self.biases["b" + str(i)]
			self.cache["Z" + str(i)] = Z
			A = sigmoid(Z)
		self.cache["A" + str(i)] = A
		np.append(self.output, A)
		return A
	
	def backward_prop(self, A, Y):
		dZ = A - Y
		m = Y.shape[1]
		for i in list(range(1, len(self.layers)))[::-1]:
			self.grads["dW" + str(i)] = np.matmul(dZ, self.cache["A" + str(i - 1)].T) / m
			self.grads["db" + str(i)] = np.sum(dZ, axis=1, keepdims=True) / m
			if i > 1:
				self.grads["dA" + str(i - 1)] = dA = np.matmul(self.weights["W" + str(i)].T, dZ)  # ?
				self.grads["dZ" + str(i - 1)] = dZ = dA * sigmoid_grad(self.cache["Z" + str(i - 1)])
		
		return dZ
	
	def compute_cost(self, A, Y):
		return np.sum(-(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / A.shape[1]
	
	def gradient_decent(self):
		for i in range(1, len(self.layers)):
			self.weights["W" + str(i)] -= self.learning_rate * self.grads["dW" + str(i)]
			self.biases["b" + str(i)] -= self.learning_rate * self.grads["db" + str(i)]
	
	def minimize(self, feed_dict, epochs=10000):
		
		self.feed_train_input(feed_dict["features"], feed_dict["labels"])
		
		# np.random.shuffle(self.train_features.T)
		for j in range(epochs):
			for i in range(1, self.train_features.shape[1] + 2 // self.batch_size):
				# print("Epoch {0}: batch {1} out of {2}".format(j,i, self.train_features.shape[1]), end="\r")
				X = self.train_features[:, (i - 1) * self.batch_size: min(i * self.batch_size, self.train_features.shape[1])]
				Y = self.train_Y[:, (i - 1) * self.batch_size: min(i * self.batch_size, self.train_Y.shape[1])]
				if len(X[0]) > 0:
					A = self.forward_prop(X)
					self.cost = np.append(self.cost, self.compute_cost(A, Y))
					self.backward_prop(A, Y)
					self.gradient_decent()
			print("Epoch {0} Training accuracy: {1} Cost: {2}.".format(j, round(self.evaluate(feed_dict), 4),self.cost[-1]))
	def predict(self, features):
		
		self.feed_test_features(features)
		
		# np.random.shuffle(self.train_features.T)
		
		output = None
		
		for i in range(1, self.train_features.shape[1] // self.batch_size + 2):
			X = self.test_features[:, (i - 1) * self.batch_size: min(i * self.batch_size, self.test_features.shape[1])]
			A = self.forward_prop(X)
			output =  A if output is None else np.hstack((output, A))
		
		return output
	
	def calc_accuracy(self, output, labels):
		self.feed_test_labels(labels)
		m = output.shape[1]
		count = 0
		
		# mask_output = (output > 0.5).astype(int)
		masked_output = np.zeros_like(output.T)
		masked_output[np.arange(output.shape[1]), output.T.argmax(1)] = 1
		masked_output = masked_output.T
		for i in range(m):
			if np.array_equal(masked_output[:, i], labels[:, i]):
				count += 1
		return count / m
	
	def evaluate(self, feed_dict):
		output = self.predict(feed_dict["features"])
		self.accuracy = self.calc_accuracy(output, feed_dict["labels"])
		return self.accuracy


def sigmoid(a):
	return 1 / (1 + np.exp(-a))


def sigmoid_grad(a):
	s = sigmoid(a)
	return s * (1 - s)
