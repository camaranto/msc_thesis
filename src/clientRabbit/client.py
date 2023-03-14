import numpy as np
from mlsocket import MLSocket
from crypto.homomorphic import Cipher
from ml.models import PrivateMLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_score, recall_score, roc_auc_score
import phe as paillier
import pickle as pkl
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA_PATH = "data/data.pkl"
ARCH =  {
        "n_features": 64, 
        "n_classes": 10,
        "hidden_layers_size": (),
        "activations": (),
        "initialization": "zeros"
    }
TRAIN = {
        'learning_rate': 0.01,
        'epochs': 10,
        'logs' : True
    }
KEY_LENGTH = 1024
PORT = 8889
NAME = "Rabbit"
class Client:
    

    def __init__(self, name, X, y, architecture):
        self.architecture = architecture
        self.name = name
        #self.__X = X
        #self.__y = y
        # self.public_key = public_key
        self.__local_model = PrivateMLPClassifier(**architecture)
        self.__federated_model = self.__local_model.copy()
        self.__X, self.X_test, self.__y, self.y_test = self.__preprocess_data(X, y)
        
    
    
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
    
    def __preprocess_data(self, X, y, test_size=0.2, scaling='standard'):
        if scaling == 'standard':
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        encoder = OneHotEncoder(sparse_output=False).fit(
                np.array([i for i in range(self.architecture['n_classes'])]).reshape(-1, 1)
            )
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        return X_train.T, X_test.T, encoder.transform(y_train.reshape(-1, 1)).T, encoder.transform(y_test.reshape(-1, 1))
    
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

    def load_data(self) -> tuple[np.ndarray, np.ndarray]:
        with open(self.DATA_PATH, 'rb') as dt_file:
            X,y = pickle.load(dt_file)
        return (X,y)
    
    def load_parameters(self) -> dict:
        with open("data/parameters.json", "r") as dt_file:
            parameters = json.load(dt_file)
        return parameters

    def start_protocol(self):

        HOST = 'anakim-dell'
        PORT = 8889
        with MLSocket() as s:
            s.connect((HOST, PORT)) 
            msg = s.recv(1024)
            print("recv", msg.decode("UTF-8"))
            s.send(bytes(self.name, encoding="UTF-8"))
            n_str = s.recv(1024)
            n = int(n_str.decode("utf-8"))
            publicKey = paillier.PaillierPublicKey(n)
            print("public key received...")
            print(n)
            print("starting local training...")
            self.local_fit(TRAIN)
            print("ended local training")


if __name__ == "__main__":
    
    with open(DATA_PATH, "rb") as f:
        X,y = pkl.load(f)
        print(X.shape, y.shape)
    capi = Client(NAME,X,y,ARCH)
    capi.start_protocol()