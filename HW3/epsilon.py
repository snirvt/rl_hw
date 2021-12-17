
import numpy as np


class SimpleEpsilon():
    def __init__(self, k, min_eps = 0):
        self.k = k
        self.min_eps = min_eps
        self.counter = 0 
    
    def update_epsilon(self):
        self.counter += 1
        return max(self.k/self.counter, self.min_eps)
    
    def __call__(self):
        return self.update_epsilon()

class LinearDecay():
    def __init__(self, eps, p, min_eps=0):
        self.eps = eps
        self.p = p
        self.min_eps = min_eps
    
    def update_epsilon(self):
        self.eps *= self.p
        return max(self.eps, self.min_eps)
    
    def __call__(self):
        return self.update_epsilon()


class ExponentialDecay():
    def __init__(self, gamma, N, min_eps=0):
        self.gamma = gamma
        self.N = N
        self.t = 0
        self.min_eps = min_eps
        
    def update_epsilon(self):
        self.t +=1
        self.eps = self.N * np.exp((-1)*self.gamma * self.t)
        return max(self.eps, self.min_eps)
    
    def __call__(self):
        return self.update_epsilon()

