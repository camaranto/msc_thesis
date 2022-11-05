import phe as paillier

from sklearn.metrics import accuracy_score

from crypto.homomorphic import Cipher
from ml.models import PrivateMLPClassifier

class Server:
    
    def __init__(self, name, key_length, n_clients, architecture):
        self.name = name
        self.model = PrivateMLPClassifier(**architecture)
        self.n_clients = n_clients
        self.public_key, self.__private_key = paillier.generate_paillier_keypair(n_length=key_length)
        
    
    def __str__(self):
        return self.name
    
    
    def decrypt_aggregate(self, encrypted_gradients):
        gradients = {}
        
        for parameter_key, parameter_value in self.model.parameters.items():
            encrypted_parameter = encrypted_gradients['d' +  parameter_key + '_enc']
            
            parameter = Cipher.decrypt_vector(self.__private_key, encrypted_parameter) / self.n_clients
            parameter = parameter.reshape(parameter_value.shape)
            
            gradients['d' + parameter_key] = parameter.copy()
            
        return gradients


class Client:
    
    def __init__(self, name, X, y, public_key, architecture):
        self.name = name
        self.__X = X
        self.__y = y
        self.public_key = public_key
        self.__local_model = PrivateMLPClassifier(**architecture)
        self.__federated_model = self.__local_model.copy()
        
    
    def __str__(self):
        return self.name
    
    
    def local_fit(self, train):
        self.__local_model.fit(self.__X, self.__y, **train)
        
        
    def local_metrics(self, X_test, y_test, encoder, error=True):
        y_pred = self.__local_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        #cross_entropy = Loss.cross_entropy(y_pred, encoder.transform(y_test.reshape(-1, 1)).T)
        
        return accuracy
    
    
    def federated_encrypted_gradient(self, sum_to=None):
        encrypted_gradient = self.__federated_model.encrypted_gradient(
            self.__X, 
            self.__y, 
            self.public_key, 
            sum_to=sum_to
        )
        
        return encrypted_gradient
    
    
    def federated_update_parameters(self, gradients, learning_rate=0.01):
        self.__federated_model.update_parameters(gradients, learning_rate)
        
    
    def federated_metrics(self, X_test, y_test):
        y_pred = self.__federated_model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy