import numpy as np
import time as time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ml.models import PrivateMLPClassifier
from simulation.entities import Client, Server

class Simulation:
    
    def __init__(self, X, y, n_clients, key_length, architecture, train, test, scaling='standard', encoding='onehot'):
        self.__n_clients = n_clients
        self.__architecture = architecture
        self.__train = train
        
        X_train, X_test, y_train, y_test = self.__preprocess_data(X, y, **test)
        
        self.__X_test = X_test.T
        self.__y_test = y_test
        
        self.__encoding = encoding
        
        self.server = Server('Central Server', key_length, n_clients, architecture)
        self.clients = self.__initialize_clients(
            X_train, 
            y_train, 
            n_clients,
            self.server.public_key, 
            architecture, 
            encoding=encoding
        )
    
    
    def __str__(self):
        return 'Simulación para {:} clientes'.format(self.__n_clients)
    
    
    def __preprocess_data(self, X, y, test_size=0.2, scaling='standard'):
        if scaling == 'standard':
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
        
        return X_train, X_test, y_train, y_test
            
    
    
    def __initialize_clients(self, X, y, n_clients, public_key, architecture, encoding='onehot'):
        clients = []
        
        n_samples = X.shape[0] // n_clients
        
        if encoding == 'onehot':
            encoder = OneHotEncoder(sparse=False).fit(
                np.array([i for i in range(architecture['n_classes'])]).reshape(-1, 1)
            )
        
        for i in range(n_clients):
            name = 'Client {:}'.format(i + 1)
            start, end = i * n_samples, (i + 1) * n_samples
            
            X_client, y_client = X[start:end, :].copy(), y[start:end].copy()
            X_client, y_client = X_client.T, encoder.transform(y_client.reshape(-1, 1)).T
            
            clients.append(Client(name, X_client, y_client, public_key, architecture))
        
        return clients
    
    
    def print_metadata(self):
        model = PrivateMLPClassifier(**self.__architecture)
        
        params = 0

        message = f'''
        ============================================================================================
        Layers size: {tuple(model.layers_size)}
        ============================================================================================
        Activations: {tuple(model.activations)}'''

        for key, value in model.parameters.items():
            params += value.shape[0] * value.shape[1]

        message = f'''{message}
        ============================================================================================
        Trainable parameters: {params}
        ============================================================================================
        '''

        print(message)
        
    
    def run_local(self):
        print('Running local training for {:d} epochs...\n'.format(self.__train['epochs']))   
        print('Metrics that each client gets on test set by training only on own local data on Test Set:\n')
        
        if self.__encoding == 'onehot':
            encoder = OneHotEncoder(sparse=False).fit(
                np.array([i for i in range(self.__architecture['n_classes'])]).reshape(-1, 1)
            )
        
        for client in self.clients:
            client.local_fit(self.__train)
            metrics = client.local_metrics(self.__X_test, self.__y_test, encoder)
            print('{:s}: {:.3%}'.format(client.name, metrics))
    
    
    def run_federated(self, logs=True):
        learning_rate, epochs = self.__train.values()
        
        print('\nRunning distributed gradient aggregation for {:d} epochs...\n'.format(epochs))
        
        start_train = time.time()
        
        for i in range(epochs):
            if logs:
                print('- Epoch {:s} / {:d}'.format(str(i + 1).zfill(len(str(epochs))), epochs), end=' : ')
            
            start_epoch = time.time()
            
            encrypted_gradients = self.clients[0].federated_encrypted_gradient(sum_to=None)
            for client in self.clients[1:]:
                encrypted_gradient = client.federated_encrypted_gradient(sum_to=encrypted_gradients)

            # Send aggregate to server and decrypt it
            gradients = self.server.decrypt_aggregate(encrypted_gradient)

            # Take gradient steps
            for client in self.clients:
                client.federated_update_parameters(gradients, learning_rate)
        
            end_epoch = time.time()
            
            if logs:
                print('Epoch Time: {:.2f}s'.format(end_epoch - start_epoch), end=', ')
                print('Total Time: {:.2f}s'.format(end_epoch - start_train))
        
        print('\nMetrics that each client gets on test set by training only on own local data on Test Set:\n')
        for client in self.clients:
            metrics = client.federated_metrics(self.__X_test, self.__y_test)
            print('{:s}: {:.3%}'.format(client.name, metrics))