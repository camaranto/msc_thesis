import numpy as np

class Cipher:
    
    @staticmethod
    def encrypt_vector(public_key, x):
        return [public_key.encrypt(i) for i in x]
    
    @staticmethod
    def decrypt_vector(private_key, x):
        return np.array([private_key.decrypt(i) for i in x])
    
    @staticmethod
    def sum_encrypted_vectors(x, y):
        if len(x) != len(y):
            raise ValueError('Encrypted vectors must have the same size')
        return [x[i] + y[i] for i in range(len(x))]