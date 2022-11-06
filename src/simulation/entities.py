import phe as paillier

from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_score, recall_score, roc_auc_score

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
        
        
    def local_metrics(self, X_test, y_test):
        y_pred = self.__local_model.predict(X_test)
        y_prob = self.__local_model.predict_probabilities(X_test)

        metrics = {
            'name': self.name,
            'accuracy': accuracy_score(y_test, y_pred),
            'loss': log_loss(y_test, y_prob.T),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'roc-auc': roc_auc_score(y_test, y_prob.T, average='weighted', multi_class='ovr')
        }

        conf_matrix = {
            'title': self.name + ' Local Confusion Matrix',
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics, conf_matrix
    
    
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
        y_prob = self.__federated_model.predict_probabilities(X_test)

        metrics = {
            'name': self.name,
            'accuracy': accuracy_score(y_test, y_pred),
            'loss': log_loss(y_test, y_prob.T),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'roc-auc': roc_auc_score(y_test, y_prob.T, average='weighted', multi_class='ovr')
        }

        conf_matrix = {
            'title': self.name + ' Federated Confusion Matrix',
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        return metrics, conf_matrix