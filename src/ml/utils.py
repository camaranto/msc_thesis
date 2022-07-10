import numpy as np

class Activation:
    
    @staticmethod
    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))
    
    @staticmethod
    def sigmoid_back(Z):
        sig = Activation.sigmoid(Z)
        return sig * (1 - sig)
    
    @staticmethod
    def tanh(Z):
        return np.tanh(Z)

    @staticmethod
    def tanh_back(Z):
        return 1 - Activation.tanh(Z) ** 2

    @staticmethod
    def relu(Z):
        return np.maximum(0, Z)

    @staticmethod
    def relu_back(Z):
        dZ = np.ones(Z.shape)
        dZ[Z <= 0] = 0
        return dZ

    @staticmethod
    def softmax(Z):
        return np.exp(Z) / np.sum(np.exp(Z), axis=0)
    
    @staticmethod
    def softmax_back(Z):
        soft = Activation.softmax(Z)
        return soft * (1 - soft)


class Assertion:
    
    @staticmethod
    def initialization_method():
        raise ValueError("Initialization method must be in ['zeros', 'random', 'he'].")
        
    @staticmethod
    def layers_activations_size(hidden_layers_size, activations):
        if len(hidden_layers_size) != len(activations):
            raise AssertionError('Hidden layers size and activations must have the same length.')
            
    @staticmethod
    def activations_values(activations):
        for activation in activations:
            if activation not in ['sigmoid', 'tanh', 'relu']:
                raise ValueError("All activations must be in ['sigmoid', 'tanh', 'relu'].")


class Initialization:
    
    @staticmethod
    def zeros(layers_size):
        n_layers, parameters = len(layers_size), {}

        for n in range(1, n_layers):
            parameters['W' + str(n)] = np.zeros((layers_size[n], layers_size[n - 1]))
            parameters['b' + str(n)] = np.zeros((layers_size[n], 1))
        
        return parameters
    
    @staticmethod
    def random(layers_size):
        n_layers, parameters = len(layers_size), {}

        for n in range(1, n_layers):
            parameters['W' + str(n)] = np.random.randn(layers_size[n], layers_size[n - 1]) * 0.01
            parameters['b' + str(n)] = np.zeros((layers_size[n], 1))
        
        return parameters
    
    @staticmethod
    def random_he(layers_size):
        n_layers, parameters = len(layers_size), {}

        for n in range(1, n_layers):
            parameters['W' + str(n)] = np.random.randn(layers_size[n], layers_size[n - 1]) / np.sqrt(layers_size[n - 1])
            parameters['b' + str(n)] = np.zeros((layers_size[n], 1))
        
        return parameters


class Loss:
    
    @staticmethod
    def cross_entropy(y, y_pred):
        return -1 * np.mean(np.sum(y * np.log(y_pred), axis=0))