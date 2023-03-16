import socket
import threading
import queue
from mlsocket import MLSocket
import numpy as np
from crypto.homomorphic import Cipher
from ml.models import PrivateMLPClassifier
import phe as paillier
import queue 
import concurrent.futures
import time
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss, precision_score, recall_score, roc_auc_score
import pickle as pkl

N_CLIENTS = 2
ARCH =  {
        "n_features": 64, 
        "n_classes": 10,
        "hidden_layers_size": (),
        "activations": (),
        "initialization": "zeros"
    }
KEY_LENGTH = 1024
PORT = 8889
class Server:

    def __init__(self, name, key_length, n_clients, architecture) -> None:
        self.name = name
        self.model = PrivateMLPClassifier(**architecture)
        self.n_clients = n_clients
        self.public_key, self.__private_key = paillier.generate_paillier_keypair(n_length=key_length)
        self.server_socket = MLSocket()


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


    def run_protocol(self, X, y) -> None:
            # get local machine name
        host = socket.gethostname()

        # bind the socket to a public host, and a well-known port
        self.server_socket.bind((host, 8889))
        print(f"listening in {host} port 8889")

        # queue up to 5 clients
        self.server_socket.listen(5)

        # create a thread pool with a maximum of 10 threads
        thread_pool = []

        # create a queue to hold client connections
        client_queue = queue.Queue()

        def handle_client(client_socket):
            try:
                # send a thank you message to the client
                message = f'Thank you for connecting to {self.name}' + "\r\n"
                client_socket.send(message.encode('ascii'))
            except socket.error:
                print("Error sending message to client")

            client_socket.close()

        while True:
            try:
                client_socket, addr = self.server_socket.accept()

                print(f'Got a connection from {addr}')

                client_queue.put(client_socket)
            except socket.error:
                print("Error accepting client connection")

            thread_pool = [t for t in thread_pool if t.is_alive()]

            if len(thread_pool) < 10:

                try:
                    client_socket = client_queue.get_nowait()
                except queue.Empty:
                    continue

                client_thread = threading.Thread(target=handle_client, args=(client_socket,))
                client_thread.start()

                thread_pool.append(client_thread)
    
    def run_protocol_threaded(self):

        host = socket.gethostname()

        # bind the socket to a public host, and a well-known port
        self.server_socket.bind((host, 8889))
        print(f"listening in {host} port 8889")

        # queue up to 5 clients
        self.server_socket.listen(5)

        def sendData(client_socket: MLSocket, data : object) -> bool:
            try:
                client_socket.send(data)
            except Exception as err:
                print(str(err))
                return False
            return True
        def recvData(client_socket):
            data = client_socket.recv(1024)
            return data
        clients = []
        while len(clients) < N_CLIENTS:
            try:
                client_socket, addr = self.server_socket.accept()
                clients.append((client_socket, addr))
                print(f'Got a connection from {addr}')
            except socket.error:
                print("Error accepting client connection")
        #print("clients:", clients)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            # Send hello
            future_to_client = {executor.submit(sendData, client_, bytes(f"HELLO from {self.name}",encoding="UTF-8")): addr_ for (client_,addr_) in clients}
            for future in concurrent.futures.as_completed(future_to_client):
                addr_ = future_to_client[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print(addr_,str(exc))
                else:
                    print(f"completed sending to {addr_}")
            # recv name
            print("recv name")
            future_to_client = {executor.submit(recvData, client_): (client_,addr_) for (client_,addr_) in clients}
            name_to_client = {}
            for future in concurrent.futures.as_completed(future_to_client):
                client_,addr_ = future_to_client[future]
                try:
                    data = future.result()
                except Exception as exc:
                    print(addr_,str(exc))
                else:
                    name_to_client[data.decode("UTF")] = (client_,addr_ )
                    print(f"completed receiving from {addr_}", data)
            print("send n")
            # send n 
            future_to_client = {executor.submit(sendData,client_,bytes(str(self.public_key.n),"utf-8")) : name_ for name_,(client_,addr_) in name_to_client.items()}
            for future in concurrent.futures.as_completed(future_to_client):
                name_ = future_to_client[future]
                try:
                    future.result()
                except Exception as err:
                    print(name_, str(err))
                else:
                    print(f"Sent public key to {name_}...")
            # recv metrics
            future_to_client = {executor.submit(recvData,client_): name_ for name_,(client_, addr_) in name_to_client.items()}
            for future in concurrent.futures.as_completed(future_to_client):
                name_ = future_to_client[future]
                try:
                    metrics = pkl.loads(future.result()) 
                except Exception as err:
                    print(name_, str(err))
                else:
                    print("=" * 5, name_ , "="*5)
                    print(metrics)
            

if __name__ == "__main__":
    server = Server("Honest Server", key_length=KEY_LENGTH, n_clients=N_CLIENTS, architecture=ARCH)
    server.run_protocol_threaded()

