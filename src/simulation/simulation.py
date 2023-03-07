import numpy as np
import pandas as pd
import time as time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from ml.models import PrivateMLPClassifier
from ml.plots import plot_confusion_matrix
from simulation.entities import Client, Server

class Simulation:
    
    def __init__(self, X, y, n_clients, key_length, architecture, train, test, scaling='standard', encoding='onehot', distribution="equal"):
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
            distribution,
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
    
    
    def __initialize_clients(self, X, y, n_clients, public_key, architecture,  distribution, encoding='onehot'):
        clients = []
        
        
        if distribution == "equal":
            n_samples = X.shape[0] // n_clients
            dist_clients = zip(range(n_clients), [n_samples]*n_clients)
        else:
            dist_clients = zip(range(n_clients), map(lambda x: int(x * X.shape[0]), distribution))
        
        if encoding == 'onehot':
            encoder = OneHotEncoder(sparse_output=False).fit(
                np.array([i for i in range(architecture['n_classes'])]).reshape(-1, 1)
            )
        acum = 0
        for i,n_samples in dist_clients:
            name = 'Client {:}'.format(i + 1)
            start, end = acum,( acum := acum + n_samples)
            
            X_client, y_client = X[start:end, :].copy(), y[start:end].copy()
            print(name, y_client.shape)
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
        
    
    def __get_local_metrics(self):
        self.__local_metrics_data, self.__local_conf_matrices = [], []
        for client in self.clients:
            metrics, conf_matrix = client.local_metrics(self.__X_test, self.__y_test)
            self.__local_metrics_data.append(metrics)
            self.__local_conf_matrices.append(conf_matrix)


    def run_local(self, output=True, plot=False):
        print('Running local training for {:d} epochs...\n'.format(self.__train['epochs']))   
        
        for client in self.clients:
            client.local_fit(self.__train)
        
        self.__get_local_metrics()

        if output:
            self.print_local_metrics(plot)


    def print_local_metrics(self, plot=False):
        df_metrics = pd.DataFrame.from_records(self.__local_metrics_data)

        print('Metrics that each client gets on Test Set by training only on own local data:\n')
        print(df_metrics.to_markdown())

        if plot:
            for conf_matrix in self.__local_conf_matrices:
                plot_confusion_matrix(**conf_matrix)
    

    def __get_federated_metrics(self):
        self.__federated_metrics_data, self.__federated_conf_matrices = [], []
        for client in self.clients:
            metrics, conf_matrix = client.federated_metrics(self.__X_test, self.__y_test)
            self.__federated_metrics_data.append(metrics)
            self.__federated_conf_matrices.append(conf_matrix)

    
    def run_federated(self, logs=True, output=True, plot=False):
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
        
        self.__get_federated_metrics()

        if output:
            self.print_federated_metrics(plot)

    
    def print_federated_metrics(self, plot=False):
        df_metrics = pd.DataFrame.from_records(self.__federated_metrics_data)

        print('\nMetrics that each client gets on Test Set by training with Federated Learning Protocol:\n')
        print(df_metrics.to_markdown())

        if plot:
            for conf_matrix in self.__federated_conf_matrices:
                plot_confusion_matrix(**conf_matrix)