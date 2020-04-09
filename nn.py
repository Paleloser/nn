import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_circles

class neural_layer():

    def __init__(self, n_in, n_n, act, d_act):
        self.act = act
        self.d_act = d_act
        self.bias = np.random.rand(1, n_n) * 2 - 1
        self.weights = np.random.rand(n_in, n_n) * 2 - 1

# Sigmoid
sig = lambda x: 1 / (1 + np.exp(-x))
d_sig = lambda x: x * (1 - x)

# Mean Squared Error
mse = lambda yi, yo: np.mean((yi - yo) ** 2)
d_mse = lambda yi, yo: yi - yo

# Creates a NN with the topology (structure of i layers with j neurons) and
# act. function given
def create_nn(topology, act, d_act):

    nn = []

    for i, layer in enumerate(topology[:-1]):
        nn.append(neural_layer(layer, topology[i + 1], act, d_act))

    return nn

# Forward pass: regular execution, data flows from the input layer trough the NN
# to the output layer.
def run_nn(nn, X):

    act_outputs = []

    for i, layer in enumerate(nn):
        layer_input = X

        if i > 0:
            layer_input = act_outputs[-1]

        layer_output = layer_input @ layer.weights + layer.bias
        act_outputs.append(layer.act(layer_output))

    return act_outputs

# Trains the given NN, with the given inputs X and expected outputs Y (therefore
# a supervised learning). Using the given cost function and learning rate (which
# by default is 0.5).
# Recommended reviewing Gradient Descent branch for more details about the learning
# rate and cost function.
def train(nn, X, Y, d_cost, lr = 0.5):

    # Firstly run the NN
    nn_act_outputs = run_nn(nn, X)
    deltas = []
    _W = []

    # Then do the Backpropagation: evaluate the results according to the expected
    for i in reversed(range(0, len(nn))):
        layer = nn[i]
        layer_inputs = X if i == 0 else nn_act_outputs[i - 1]
        layer_outputs = nn_act_outputs[i]

        # Last layer backpropagation
        if i == len(nn) - 1:
            deltas.insert(0, d_cost(layer_outputs, Y) * layer.d_act(layer_outputs))
        # Cost effect for the rest of the layers
        else:
            deltas.insert(0, deltas[0] @ _W.T * layer.d_act(layer_outputs))

        _W = layer.weights

        # Gradient descent:
        layer.bias = layer.bias - np.mean(deltas[0], axis=0, keepdims=True) * lr
        layer.weights = layer.weights - layer_inputs.T @ deltas[0] * lr

# Dataset initialization
n = 500  # Number of dots
p = 2  # Dimensions

# Below we make 2 circles:
# X will contain X and Y for each dot => [[x0, y0], [x1, y1], ..., [xN-1, YN-1]]
# Y will tell wether a dot is from the inner circle or the outer one [1, 0, ..., 0]
X, Y = make_circles(n, factor=0.5, noise=0.05)
Y = Y[:, np.newaxis]

topology = [p, 8, 8, 1]
nn = create_nn(topology, sig, d_sig)

loss = []
epochs = 2500

for i in range(epochs):

    train(nn, X, Y, d_mse, lr=0.01)

    # Take a sample of the current trainning status.
    if i % 10 == 0:
        # 1. Run the NN (the [-1] in python means the last item ov the array,
        # note that run_nn returns an array with the ouputs of all layers, as
        # it is used by training() to do the backpropagation on each layer)
        y_i = run_nn(nn, X)[-1]
        #2. Sample the error
        loss.append(mse(y_i, Y))

res = 50

_x0 = np.linspace(-1.5, 1.5, res)
_x1 = np.linspace(-1.5, 1.5, res)
_Y = np.zeros((res, res))

for i0, x0 in enumerate(_x0):
    for i1, x1 in enumerate(_x1):
        _Y[i0, i1] = run_nn(nn, np.array([[x0, x1]]))[-1][0][0]

plt.pcolormesh(_x0, _x1, _Y, cmap="coolwarm")
plt.scatter(X[Y[:, 0] == 0, 0], X[Y[:, 0] == 0, 1], c='skyblue')
plt.scatter(X[Y[:, 0] == 1, 0], X[Y[:, 0] == 1, 1], c='salmon')
plt.axis('equal')
plt.show()
plt.plot(range(len(loss)), loss)
plt.show()
