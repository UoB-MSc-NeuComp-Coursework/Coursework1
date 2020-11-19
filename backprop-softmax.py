
# Neural Computation (Extended)
# CW1: Backpropagation and Softmax
# Autumn 2020
#

import numpy as np
import time
import fnn_utils
import matplotlib.pyplot as plt
from tqdm import tqdm #bar showing
# Some activation functions with derivatives.
# Choose which one to use by updating the variable phi in the code below.

def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_d(x):
    s = sigmoid(x)
    return s * (1-s)
def relu(x):
    return x * (x > 0)
def relu_d(x):
    return 1 * (x > 0)
def swish(x):
    sig = sigmoid(x)
    return x * sig
def swish_d(x):
    swi = swish(x)
    sig = sigmoid(x)
    return swi + sig*(1-swi)
def tanh(x):
    return 2/(1+np.exp(-2*x)) - 1
def tanh_d(x):
    t = tanh(x)
    return 1-(t)**2
def linear(x):
    return x
def linear_d(x):
    return 1
class BackPropagation:

    # The network shape list describes the number of units in each
    # layer of the network. The input layer has 784 units (28 x 28
    # input pixels), and 10 output units, one for each of the ten
    # classes.

    def __init__(self,network_shape=[784, 20 , 20, 20, 10]):  #network_shape=(5)[784,20,20,20,10]   (8)[784,20,20,20, 20, 20, 20, 10]

        # Read the training and test data using the provided utility functions
        self.trainX, self.trainY, self.testX, self.testY = fnn_utils.read_data()

        # Number of layers in the network
        self.L = len(network_shape)

        self.crossings = [(1 if i < 1 else network_shape[i-1],network_shape[i]) for i in range(self.L)]

        # Create the network
        self.a             = [np.zeros(m) for m in network_shape]
        self.db            = [np.zeros(m) for m in network_shape]
        self.b             = [np.random.normal(0,1/10,m) for m in network_shape]
        self.z             = [np.zeros(m) for m in network_shape]
        self.delta         = [np.zeros(m) for m in network_shape]
        self.w             = [np.random.uniform(-1/np.sqrt(m0),1/np.sqrt(m0),(m1,m0)) for (m0,m1) in self.crossings]
        self.dw            = [np.zeros((m1,m0)) for (m0,m1) in self.crossings]
        self.nabla_C_out   = np.zeros(network_shape[-1])

        # Choose activation function
        self.phi           = relu
        self.phi_d         = relu_d
        
        # Store activations over the batch for plotting
        self.batch_a       = [np.zeros(m) for m in network_shape]
                
    def forward(self, x):
        """ Set first activation in input layer equal to the input vector x (a 24x24 picture), 
            feed forward through the layers, then return the activations of the last layer.
        """

        inp = (x - np.amin(x)) / (np.amax(x) - np.amin(x))
        self.a[0] = inp - 0.5  # Center the input values between [-0.5,0.5]
        for i in range(1, self.L - 1):
            self.z[i] = np.dot(self.w[i], self.a[i-1]) + self.b[i]
            self.a[i] = self.phi(self.z[i])
            
        self.z[self.L - 1] = np.dot(self.w[self.L - 1], self.a[self.L - 2]) + self.b[self.L - 1]
        self.a[self.L - 1] = self.softmax(self.z[self.L - 1])
        
        return(self.a[self.L-1])

    def softmax(self, z):

        return np.exp(z) / sum(np.exp(z))

    def loss(self, pred, y):
        return -np.log(pred[np.argmax(y)])

    
    def backward(self,x, y):
        """ Compute local gradients, then return gradients of network.
        """
        '''
        self.delta = self.a[self.L-1] - y  # computes the local gradients for each layer of the network.
        self.dw = np.dot(self.delta, np.transpose(self.a[self.L-1]))  # computes the local gradients with respect to the weights.
        self.db = self.delta  # computes the local gradients with respect to biases.
        '''
        network_shape = [784, 20, 20, 20, 20, 10]
        output = self.forward(x)
        self.delta[self.L - 1] = output - y
        self.dw[self.L - 1] = np.dot(self.delta[self.L - 1].reshape(np.size(self.b[self.L - 1]), 1),
                            self.a[self.L - 2].reshape(1, np.size(self.b[self.L - 2])))  # (10,20)
        self.db[self.L - 1] = self.delta[self.L - 1]  # (10,)

        for j in range(self.L - 2, 0, -1):
            self.delta[j] = np.dot(self.delta[j+1], self.w[j+1]) * self.phi_d(self.z[j])  # (20,)
            self.dw[j] = np.dot(self.delta[j].reshape(np.size(self.b[j]), 1),
                            self.a[j-1].reshape(1, np.size(self.b[j-1])))  # (20,20)
            self.db[j] = self.delta[j]
        return self.dw, self.db  # returns the gradients of the network.

    # Return predicted image class for input x
    def predict(self, x):
        y_pred = np.argmax(self.forward(x))
        return y_pred

    # Return predicted percentage for class j
    def predict_pct(self, j):
        return self.a[self.L - 1][j]
    
    def evaluate(self, X, Y, N):
        """ Evaluate the network on a random subset of size N. """
        num_data = min(len(X),len(Y))
        samples = np.random.randint(num_data,size=N)
        results = [(self.predict(x), np.argmax(y)) for (x,y) in zip(X[samples],Y[samples])]
        return sum(int(x==y) for (x,y) in results)/N

    
    def sgd(self,
            batch_size=50,
            epsilon=0.01,
            epochs=150):

        """ Mini-batch gradient descent on training data.

            batch_size: number of training examples between each weight update
            epsilon:    learning rate
            epochs:     the number of times to go through the entire training data
        """
        
        # Compute the number of training examples and number of mini-batches.
        N = min(len(self.trainX), len(self.trainY))
        num_batches = int(N/batch_size)

        # Variables to keep track of statistics
        loss_log      = []
        test_acc_log  = []
        train_acc_log = []

        timestamp = time.time()
        timestamp2 = time.time()

        predictions_not_shown = True
        
        # In each "epoch", the network is exposed to the entire training set.
        for t in range(epochs):

            # We will order the training data using a random permutation.
            permutation = np.random.permutation(N)
            
            # Evaluate the accuracy on 1000 samples from the training and test data
            test_acc_log.append(self.evaluate(self.testX, self.testY, 1000))
            train_acc_log.append(self.evaluate(self.trainX, self.trainY, 1000))
            batch_loss = 0

            print('\n Epochs'+str(t)+'  Test acc: '+str(test_acc_log[t])+' Train acc: ' +str(train_acc_log[t])+' Max Test acc: '+ str(max(test_acc_log))+' Max Train acc: '+str(max(train_acc_log)))

            for k in tqdm(range(num_batches)):
                # Reset buffer containing updates
                nabla_b = [np.zeros(b.shape) for b in self.b]
                nabla_w = [np.zeros(w.shape) for w in self.w]
                
                # Mini-batch loop
                for i in range(batch_size):

                    # Select the next training example (x,y)
                    x = self.trainX[permutation[k*batch_size+i]]
                    y = self.trainY[permutation[k*batch_size+i]]

                    # Feed forward inputs
                    self.a[self.L - 1] = self.forward(x)

                    
                    # Compute gradients
                    dw, db = self.backward(x, y)[0], self.backward(x, y)[1]

                    nabla_b = [n_b + d_b for n_b, d_b in zip(nabla_b, db)]
                    nabla_w = [n_w + d_w for n_w, d_w in zip(nabla_w, dw)]

                    # Update loss log
                    batch_loss += self.loss(self.a[self.L-1], y)

                    for l in range(self.L):
                        self.batch_a[l] += self.a[l] / batch_size
                                    
                # Update the weights at the end of the mini-batch using gradient descent
                for l in range(1,self.L):
                    self.w[l] = self.w[l] - (epsilon * (nabla_w[l]/batch_size))
                    self.b[l] = self.b[l] - (epsilon * (nabla_b[l]/batch_size))
                
                # Update logs
                loss_log.append( batch_loss / batch_size )
                batch_loss = 0

                # Update plot of statistics every 10 seconds.
                if time.time() - timestamp > 10:
                    timestamp = time.time()
                    fnn_utils.plot_stats(self.batch_a,
                                         loss_log,
                                         test_acc_log,
                                         train_acc_log)

                # Display predictions every 20 seconds.
                if (time.time() - timestamp2 > 20) or predictions_not_shown:
                    predictions_not_shown = False
                    timestamp2 = time.time()
                    fnn_utils.display_predictions(self,show_pct=True)

                # Reset batch average
                for l in range(self.L):
                    self.batch_a[l].fill(0.0)



# Start training with default parameters.

def main():
    bp = BackPropagation()
    start = time.time()
    bp.sgd()
    print('train accuracy: ' + str(bp.evaluate(bp.trainX, bp.trainY, 6000)))
    print('test accuracy: ' + str(bp.evaluate(bp.testX, bp.testY, 1000)))
    print('time(s): ' + str(time.time() - start))
    print('time(m): ' + str((time.time() - start) / 60))
    plt.show(block=True)
if __name__ == "__main__":
    main()
    
