import tensorflow as tf
import numpy as np

class sign_classifier:
    def initialize_parameters(self, layer_dims):
        self.parameters=dict()
        for l in range(1, len(layer_dims)//2):
            self.parameters["W"+str(l)]=tf.Variable(np.random.randn(layer_dims[l],layer_dims[l-1])*np.sqrt(2/layer_dims[l-1]))
            self.parameters["b"+str(l)]=tf.Variable(np.zeros((layer_dims[l],1)))
    def forward(self):
        return cache