import numpy as np

class Cipher:
    
    @staticmethod
    def encrypt_vector(public_key, x):
        return [public_key.encrypt(i) for i in x]
    
    @staticmethod
    def decrypt_vector(private_key, x):
        return np.array([private_key.decrypt(i) for i in x])
    
    @staticmethod
    def sum_encrypted_vectors(*x):
        if len(x) < 2:
            raise ValueError('Must provide at least two vectors')
        x_size = len(x[0])
        if not all([ len(x_) == x_size for x_ in x]):
            raise ValueError('Encrypted vectors must have the same size')
        acum_encrypted_vector = list()
        for i in range(x_size):    
            acum = 0
            for x_ in x:
                acum += x_[i]
            acum_encrypted_vector.append(acum)
        return acum_encrypted_vector