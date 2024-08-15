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
        
    def forward(self, X):
        Al=X
        for l in range(1, self.nlayers-1):
            Zl=tf.math.add(tf.linalg.matmul(self.parameters["W"+str(l)], Al),self.parameters["b"+str(l)])
            Al=tf.keras.activations.relu(Zl)
        return tf.math.add(tf.linalg.matmul(self.parameters["W"+str(self.nlayers-1)], Al),self.parameters["b"+str(self.nlayers-1)])
    
    def compute_total_loss(self, y_pred, y_true, l2_lambda):
        cross_entropy_loss = tf.reduce_sum(
            tf.keras.losses.categorical_crossentropy(y_true,y_pred, from_logits=True)
        )
        l2_loss = 0
        for l in range(1, self.nlayers):
            l2_loss += tf.nn.l2_loss(self.parameters["W" + str(l)])
        
        # Total loss: cross-entropy + L2 regularization term
        total_loss = cross_entropy_loss + l2_lambda * l2_loss
        
        return total_loss

    def training(self, train, test, hidden_nodes, learning_rate, minibatch_size, num_epochs, l2_lambda):
        print(f"Initializing training with hyperparameters : hidden_nodes:{hidden_nodes}; learning_rate:{learning_rate}; minibatch_size:{minibatch_size}; l2_lambda:{l2_lambda}")
        costs = []
        train_acc = []
        test_acc = []

        layer_dims=[64**2]
        for layer in hidden_nodes:
            layer_dims.append(layer)
        layer_dims.append(24)
        
        self.initialize_parameters(layer_dims)
        optimizer = tf.keras.optimizers.Adam(learning_rate)
        
        # The CategoricalAccuracy will track the accuracy for this multiclass problem
        test_accuracy = tf.keras.metrics.CategoricalAccuracy()
        train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        
        # We can get the number of elements of a dataset using the cardinality method
        m = train.cardinality().numpy()
        
        minibatches = train.batch(minibatch_size).prefetch(8)
        test_minibatches = test.batch(minibatch_size).prefetch(8)

        try:
            for epoch in range(num_epochs):

                epoch_total_loss = 0.
                
                #We need to reset object to start measuring from 0 the accuracy each epoch
                train_accuracy.reset_state()
                
                for (minibatch_X, minibatch_Y) in minibatches:
                    
                    with tf.GradientTape() as tape:
                        ZL = self.forward(tf.transpose(minibatch_X))

                        minibatch_total_loss = self.compute_total_loss(tf.transpose(ZL), minibatch_Y, l2_lambda)

                    # We accumulate the accuracy of all the batches
                    train_accuracy.update_state(minibatch_Y, tf.transpose(ZL))
                    
                    #We compute the gradients and update the weights with Adam
                    trainable_variables = [v for v in self.parameters.values()]
                    grads = tape.gradient(minibatch_total_loss, trainable_variables)
                    optimizer.apply_gradients(zip(grads, trainable_variables))
                    epoch_total_loss += minibatch_total_loss
                
                # We divide the epoch total loss over the number of samples and compute the test accuracy
                if epoch%10==0:
                    epoch_total_loss /= m
                    for (minibatch_X, minibatch_Y) in test_minibatches:
                        ZL = self.forward(tf.transpose(minibatch_X))
                        test_accuracy.update_state(minibatch_Y, tf.transpose(ZL))

                    print(f"Epoch {epoch}: cost: {epoch_total_loss}; train_acc: {train_accuracy.result().numpy()}; test_acc: {test_accuracy.result().numpy()}")
                    
                    costs.append(epoch_total_loss)
                    train_acc.append(train_accuracy.result())
                    test_acc.append(test_accuracy.result())
                    test_accuracy.reset_state()
            costs.append(epoch_total_loss)
            train_acc.append(train_accuracy.result())
            test_acc.append(test_accuracy.result())
        except KeyboardInterrupt:
            print("Keyboard pressed")
        
        return self.parameters, costs, train_acc, test_acc