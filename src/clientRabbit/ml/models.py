import numpy as np

from crypto.homomorphic import Cipher
from ml.utils import Activation, Assertion, Initialization, Loss

class MLPClassifier:
    
    def __init__(self, n_features, n_classes, hidden_layers_size=(), activations=(), initialization='he'):
        Assertion.layers_activations_size(hidden_layers_size, activations)
        Assertion.activations_values(activations)
        
        self.n_features = n_features
        self.n_classes = n_classes
        self.layers_size = self.__get_layers_size(n_features, hidden_layers_size, n_classes)
        self.activations = self.__get_activations(activations)
        
        if initialization == 'zeros':
            self.parameters = Initialization.zeros(self.layers_size)
        
        elif initialization == 'random':
            self.parameters = Initialization.random(self.layers_size)
        
        elif initialization == 'he':
            self.parameters = Initialization.random_he(self.layers_size)
        
        else:
            Assertion.initialization_method()
            
        
    def __get_layers_size(self, n_features, hidden_layers_size, n_classes):
        layers_size = [n_features]
        layers_size.extend(hidden_layers_size)
        layers_size.append(n_classes)
        
        return layers_size
    
    
    def __get_activations(self, activations):
        activations = list(activations)
        activations.append('softmax')
        
        return activations
    
    
    def __forward_activation(self, A_prev, W, b, activation):
        Z = np.dot(W, A_prev) + b
        
        if activation == 'sigmoid':
            A = Activation.sigmoid(Z)
            
        elif activation == 'tanh':
            A = Activation.tanh(Z)
            
        elif activation == 'relu':
            A = Activation.relu(Z)
            
        elif activation == 'softmax':
            A = Activation.softmax(Z)
            
        return (A, Z)
    
    
    def __backward_activation(self, Z, activation):
        if activation == 'sigmoid':
            dA = Activation.sigmoid_back(Z)
            
        elif activation == 'tanh':
            dA = Activation.tanh_back(Z)
            
        elif activation == 'relu':
            dA = Activation.relu_back(Z)
            
        elif activation == 'softmax':
            dA = Activation.softmax_back(Z)
            
        return dA
    
    
    def __backpropagation(self, y, y_pred, cache):
        n_layers, gradients, m = len(self.layers_size) - 1, {}, y_pred.shape[1]
        
        dA = -(np.divide(y, y_pred) - np.divide(1 - y, 1 - y_pred))
        
        for n in reversed(range(n_layers)):
            dZ = dA * self.__backward_activation(cache['Z' + str(n + 1)], self.activations[n])
            
            gradients['dW' + str(n + 1)] = 1 / m * np.dot(dZ, cache['A' + str(n)].T)
            gradients['db' + str(n + 1)] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            
            dA = np.dot(self.parameters['W' + str(n + 1)].T, dZ)
        
        return gradients
    
    
    def predict_probabilities(self, X, cache=False):
        A, n_layers = X, len(self.layers_size) - 1
        
        cache_back = {
            'A0': A
        }
        
        for n in range(n_layers):
            A, Z = self.__forward_activation(
                A, 
                self.parameters['W' + str(n + 1)], 
                self.parameters['b' + str(n + 1)], 
                self.activations[n]
            )
            
            cache_back['Z' + str(n + 1)] = Z
            cache_back['A' + str(n + 1)] = A
            
        return (A, cache_back) if cache else A
    
    
    def predict(self, X):
        y_prob = self.predict_probabilities(X)
        
        return np.argmax(y_prob, axis=0).reshape(-1,)
    
    
    def compute_gradient(self, X, y, logs=False):
        y_pred, cache = self.predict_probabilities(X, cache=True)
        
        if logs:
            print('Loss:', Loss.cross_entropy(y, y_pred))
        
        gradients = self.__backpropagation(y, y_pred, cache)
        
        return gradients
    
    
    def update_parameters(self, gradients, learning_rate=0.01):
        n_layers = len(self.layers_size) - 1
        
        for n in range(n_layers):
            self.parameters['W' + str(n + 1)] -= learning_rate * gradients['dW' + str(n + 1)]
            self.parameters['b' + str(n + 1)] -= learning_rate * gradients['db' + str(n + 1)]
            
    
    def fit(self, X, y, learning_rate=0.01, epochs=50, logs=False):
        for _ in range(epochs):
            print("epoch : ", _, end=" ")
            gradients = self.compute_gradient(X, y, logs=logs)
            self.update_parameters(gradients, learning_rate)
            
            
    def copy(self):
        model_copy = self.__class__(self.n_features, self.n_classes, self.layers_size[1:-1], self.activations[:-1])
        
        model_copy.parameters = self.parameters
        
        return model_copy


class PrivateMLPClassifier(MLPClassifier):
    
    def __init__(self, n_features, n_classes, hidden_layers_size=(), activations=(), initialization='he'):
        
        super().__init__(
            n_features, 
            n_classes, 
            hidden_layers_size=hidden_layers_size, 
            activations=activations, 
            initialization=initialization
        )
        
    
    def encrypted_gradient(self, X, y, public_key, sum_to=None, logs=False):
        encrypted_gradients = {}
        gradients = self.compute_gradient(X, y, logs=logs)
        
        for gradient_key, gradient_value in gradients.items():
            gradient_value = gradient_value.reshape(gradient_value.shape[0] * gradient_value.shape[1],)
            
            encrypted_gradients[gradient_key + '_enc'] = Cipher.encrypt_vector(public_key, gradient_value)
            
            if sum_to is not None:
                encrypted_gradients[gradient_key + '_enc'] = Cipher.sum_encrypted_vectors(
                    sum_to[gradient_key + '_enc'], 
                    encrypted_gradients[gradient_key + '_enc']
                )
        
        return encrypted_gradients