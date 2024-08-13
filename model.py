import tensorflow as tf
import numpy as np

class sign_classifier:
    def initialize_parameters(self, layer_dims):
        initializer = tf.keras.initializers.GlorotNormal(seed=1)
        self.nlayers = len(layer_dims)
        
        self.parameters=dict()
        for l in range(1, self.nlayers):
            self.parameters["W"+str(l)]=tf.Variable(initializer(shape=(layer_dims[l], layer_dims[l-1])), dtype=tf.float32, trainable=True)
            self.parameters["b"+str(l)]=tf.Variable(initializer(shape=(layer_dims[l],1)), dtype=tf.float32, trainable=True)

        """W1 = tf.Variable(initializer(shape=(layer_dims[1], layer_dims[0])), dtype=tf.float32, trainable=True)
        b1 = tf.Variable(initializer(shape=(layer_dims[1],1)), dtype=tf.float32, trainable=True)
        W2 = tf.Variable(initializer(shape=(layer_dims[2], layer_dims[1])), dtype=tf.float32, trainable=True)
        b2 = tf.Variable(initializer(shape=(layer_dims[2],1)), dtype=tf.float32, trainable=True)
        W3 = tf.Variable(initializer(shape=(layer_dims[3], layer_dims[2])), dtype=tf.float32, trainable=True)
        b3 = tf.Variable(initializer(shape=(layer_dims[3],1)), dtype=tf.float32, trainable=True)
        
        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2,
                      "W3": W3,
                      "b3": b3}
        return parameters"""
        
    def forward(self, X):
        Al=X
        for l in range(1, self.nlayers-1):
            Zl=tf.math.add(tf.linalg.matmul(self.parameters["W"+str(l)], Al),self.parameters["b"+str(l)])
            Al=tf.keras.activations.relu(Zl)
        return tf.math.add(tf.linalg.matmul(self.parameters["W"+str(self.nlayers-1)], Al),self.parameters["b"+str(self.nlayers-1)])
        """W1 = parameters['W1']
        b1 = parameters['b1']
        W2 = parameters['W2']
        b2 = parameters['b2']
        W3 = parameters['W3']
        b3 = parameters['b3']
        
        Z1 = tf.math.add(tf.linalg.matmul(W1, X),b1)
        A1 = tf.keras.activations.relu(Z1)
        Z2 = tf.math.add(tf.linalg.matmul(W2, A1),b2)
        A2 = tf.keras.activations.relu(Z2)
        Z3 = tf.math.add(tf.linalg.matmul(W3, A2),b3)
        return Z3"""
    
    def compute_total_loss(self, y_pred, y_true):
        return tf.reduce_sum(
            tf.keras.losses.categorical_crossentropy(y_true,y_pred, from_logits=True)
        )

    def training(self, X_train, Y_train, X_test, Y_test, learning_rate = 0.0001, num_epochs = 10, minibatch_size = 64):
        costs = []
        train_acc = []
        test_acc = []

        layer_dims = [784, 64, 32, 24]
        self.initialize_parameters(layer_dims)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # The CategoricalAccuracy will track the accuracy for this multiclass problem
        test_accuracy = tf.keras.metrics.CategoricalAccuracy()
        train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        
        dataset = tf.data.Dataset.zip((X_train, Y_train))
        test_dataset = tf.data.Dataset.zip((X_test, Y_test))
        
        # We can get the number of elements of a dataset using the cardinality method
        m = dataset.cardinality().numpy()
        
        minibatches = dataset.batch(minibatch_size).prefetch(8)
        test_minibatches = test_dataset.batch(minibatch_size).prefetch(8)

        try:
            for epoch in range(num_epochs):

                epoch_total_loss = 0.
                
                #We need to reset object to start measuring from 0 the accuracy each epoch
                train_accuracy.reset_state()
                
                for (minibatch_X, minibatch_Y) in minibatches:
                    
                    with tf.GradientTape() as tape:
                        ZL = self.forward(tf.transpose(minibatch_X))

                        minibatch_total_loss = self.compute_total_loss(tf.transpose(ZL), minibatch_Y)

                    # We accumulate the accuracy of all the batches
                    train_accuracy.update_state(minibatch_Y, tf.transpose(ZL))
                    
                    #We compute the gradients and update the weights with Adam
                    trainable_variables = [v for v in self.parameters.values()]
                    grads = tape.gradient(minibatch_total_loss, trainable_variables)
                    optimizer.apply_gradients(zip(grads, trainable_variables))
                    epoch_total_loss += minibatch_total_loss
                
                # We divide the epoch total loss over the number of samples and compute the test accuracy
                epoch_total_loss /= m
                for (minibatch_X, minibatch_Y) in test_minibatches:
                    ZL = self.forward(tf.transpose(minibatch_X))
                    test_accuracy.update_state(minibatch_Y, tf.transpose(ZL))

                print(f"Epoch {epoch}: cost: {epoch_total_loss}; train_acc: {train_accuracy.result().numpy()}; test_acc: {test_accuracy.result().numpy()}")
                
                costs.append(epoch_total_loss)
                train_acc.append(train_accuracy.result())
                test_acc.append(test_accuracy.result())
                test_accuracy.reset_state()
        except KeyboardInterrupt:
            print("Keyboard pressed")

        return self.parameters, costs, train_acc, test_acc