import random
import math

# Referenced: https://github.com/mattm/simple-neural-network
class Neuron:
	def __init__(self, bias):
		self.bias = bias
		self.weights = []

	def calculate_output(self, inputs):
		self.inputs = inputs
		self.output = self.sigmoid(self.calculate_net_input())
		return self.output


	def calculate_net_input(self):
		total = 0
		for i in range(len(self.inputs)):
			total += self.inputs[i] * self.weights[i]
		return total + self.bias

	def sigmoid(self, net_input):
		return 1 / (1 + math.exp(-net_input))

	def relu(self, net_input):
		return max(0, net_input)

	def least_squares_error(self, target):
		return 0.5 * (target - self.output) ** 2

	# What's next?
	# Need to determine how much the neuron's total input has to change to move closer to the expected output
	# What is this value?
	# ∂E/∂zⱼ = ∂E/∂yⱼ * dyⱼ/dzⱼ

	def derivative_error_wrt_net_input(self, target):
		return self.derivative_error_wrt_output(target) * self.derivative_net_input_wrt_input()

	# Least Squares Error: 1/2 * (tⱼ - yⱼ)^2
	# The partial derivate of the error with respect to actual output then is calculated by:
	# ∂E/∂yⱼ = -(tⱼ - yⱼ)
	def derivative_error_wrt_output(self, target):
		return -1 * (target - self.output)

	# dyⱼ/dzⱼ = yⱼ * (1 - yⱼ)
	def derivative_net_input_wrt_input(self):
		return self.output * (1 - self.output)

	# The total net input is the weighted sum of all the inputs to the neuron and their respective weights:
	# zⱼ = netⱼ = x₁w₁ + x₂w₂ ...
    # The partial derivative of the total net input with respective to a given weight (with everything else held constant) then is:
    # = ∂zⱼ/∂wᵢ = some constant + 1 * xᵢw₁^(1-0) + some constant ... = xᵢ
    def derivative_total_net_input_wrt_weight(self, index):
    	return self.inputs[index]



class Layer:
	"""
	Creates a Layer in the network consisting of neurons/units.
	Each layer has one bias.


	Parameters:
	------------
	N : Integer
		Number of neurons/units in the layer

	bias : Float
		bias of the layer

	neurons : List of objects of class Neuron
	"""

	def __init__(self, N, bias):
		if bias:
			self.bias = bias
		else:
			self.bias = random.random()

		self.neurons = []

		for i in range(N):
			self.neurons.append(self.bias)

    def feed_forward(self, inputs):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def get_outputs(self):
        outputs = []
        for neuron in self.neurons:
            outputs.append(neuron.output)
        return outputs

class NeuralNetwork:
	learning_rate = 0.5

	def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_bias = None, output_layer_weights = None):
		self.num_inputs = num_inputs
		self.hidden_layer = Layer(num_hidden, hidden_layer_bias)
		self.output_layer = Layer(num_outputs, output_layer_bias)



	def initialize_weights_hidden_layer(self, hidden_layer_weights):
		weight_num = 0
		for h in range(len(self.hidden_layer.neurons)):
			for i in range(self.num_inputs):
				if hidden_layer_weights:
					self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
				else:
					self.hidden_layer.neurons[h].weights.append(random.random())
				weight_num += 1


	def initialize_weights_output_layer(self, output_layer_weights):
		weight_num = 0
		for h in range(len(self.output_layer.neurons)):
			for i in range(self.num_inputs):
				if output_layer_weights:
					self.output_layer.neurons[h].weights.append(output_layer_weights[weight_num])
				else:
					self.output_layer.neurons[h].weights.append(random.random())
				weight_num += 1		

	def feed_forward(self, inputs):
		hidden_layer_outputs = self.hidden_layer.feed_forward(inputs)
		return self.output_layer.feed_forward(hidden_layer_outputs)

	# Using Stochastic Gradient Descent
	# Parameter "inputs" is the input to the network.
	# Parameter "target" is the groundtruth label for the given output.
	def train(self, inputs, target):
		self.feed_forward(inputs)

		# Step 1: Calculate output neurons derivative 
		# Calculate ∂E/∂z
		derivative_error_wrt_output_layer_net_input = [0] * len(self.output_layer.neurons)
		for i in range(len(self.output_layer.neurons)):
			#  Calculate ∂E/∂zⱼ
			derivative_error_wrt_output_layer_net_input[i] = self.output_layer.neurons[i].derivative_error_wrt_net_input(target[i])

		# Step 2: Calculate hidden neurons derivative
		# dE/dyⱼ = Σ ∂E/∂zⱼ * ∂z/∂yⱼ = Σ ∂E/∂zⱼ * wᵢⱼ
		derivative_error_wrt_hidden_layer_net_input = [0] * len(self.hidden_layer.neurons)
		for i in range(len(self.hidden_layer.neurons)):
			temp = 0
			for j in range(len(self.output_layer.neurons)):
				# what is temp?
				# 
				temp += derivative_error_wrt_output_layer_net_input[j] * self.output_layer.neurons[j].weights[i]

			derivative_error_wrt_hidden_layer_net_input[i] = temp * self.hidden_layer.neurons[i].derivative_net_input_wrt_input()

		# Step 3: Update output neurons weights
		for o in range(len(self.output_layer.neurons)):
			for weight_num in range(len(self.output_layer.neurons[o].weights)):
				# ∂Eⱼ/∂wᵢⱼ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢⱼ
				derivative_error_wrt_weight = derivative_error_wrt_output_layer_net_input[o] * self.output_layer.neurons[o].derivative_total_net_input_wrt_weight(weight_num)
				# Δw = α * ∂Eⱼ/∂wᵢ
				self.output_layer.neurons[o].weights[weight_num] -= self.learning_rate * derivative_error_wrt_weight

		# Step 4: Update hidden neuron weights
		for h in range(len(self.hidden_layer.neurons)):
			for weight_num in range(len(self.hidden_layer.neurons[h].weights)):
				# ∂Eⱼ/∂wᵢ = ∂E/∂zⱼ * ∂zⱼ/∂wᵢ
				derivative_error_wrt_weight = derivative_error_wrt_hidden_layer_net_input[h] * self.hidden_layer.neurons[h].derivative_total_net_input_wrt_weight(weight_num)
				self.hidden_layer.neurons[h].weights[weight_num] -= self.learning_rate * derivative_error_wrt_weight

	def calculate_total_error(self, training_data):
		total_error = 0
		for t in range(len(training_data)):
			training_x, training_output = training_data[t]
			self.feed_forward(training_x)

			for o in range(len(training_output)):
				total_error += self.output_layer.neurons[o].calculate_error(training_output[o])
		return total_error

