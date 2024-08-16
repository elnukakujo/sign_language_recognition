import tensorflow as tf
from display import save_plot_html
import json
import os

class sign_classifier:
    def __init__(self, model_name):
        self.model_name=model_name
        self.save_path=f"data/models/{self.model_name}/"
    
    def set_hyperparameters(self, hyperparameters, input_shape=False, output_shape=False, new_hyperparameters=False):
        if new_hyperparameters:
            for key, value in new_hyperparameters.items():
                hyperparameters[key].append(value[0])
        self.hyperparameters=hyperparameters
        self.layer_dims=[input_shape]+self.hyperparameters["hidden_nodes"][-1]+[output_shape]
        print("Hyperparameters defined")
        return True
    
    def get_hyperparameters(self):
        return (
            self.hyperparameters["lr"][-1],
            self.hyperparameters["minibatch_size"][-1],
            self.hyperparameters["l2_lambda"][-1]
        )
    
    def initialize_parameters(self):
        initializer = tf.keras.initializers.GlorotNormal(seed=1)
        self.parameters=dict()
        for l in range(1, len(self.layer_dims)):
            self.parameters["W"+str(l)]=tf.Variable(initializer(shape=(self.layer_dims[l], self.layer_dims[l-1])), dtype=tf.float32, trainable=True)
            self.parameters["b"+str(l)]=tf.Variable(initializer(shape=(self.layer_dims[l],1)), dtype=tf.float32, trainable=True)
        print("Parameters initialized")
        return True
        
    def forward(self, X):
        Al=X
        nlayers = len(self.layer_dims)
        for l in range(1, nlayers-1):
            Zl=tf.math.add(tf.linalg.matmul(self.parameters["W"+str(l)], Al),self.parameters["b"+str(l)])
            Al=tf.keras.activations.relu(Zl)
        return tf.math.add(tf.linalg.matmul(self.parameters["W"+str(nlayers-1)], Al),self.parameters["b"+str(nlayers-1)])
    
    def compute_total_loss(self, y_pred, y_true, l2_lambda):
        cross_entropy_loss = tf.reduce_sum(
            tf.keras.losses.categorical_crossentropy(y_true,y_pred, from_logits=True)
        )
        l2_loss = 0
        for l in range(1, len(self.layer_dims)):
            l2_loss += tf.nn.l2_loss(self.parameters["W" + str(l)])
        
        # Total loss: cross-entropy + L2 regularization term
        total_loss = cross_entropy_loss + l2_lambda * l2_loss
        
        return total_loss
    
    def save_results(self, plot=False):
        path=self.save_path+"hypertuning/"
        
        folder_to_create = [path, path+"hyper_parameters/", path+"metrics/", path+"plots/"]
        for folder in folder_to_create:
            if not os.path.exists(folder):
                os.mkdir(folder)
        
        with open(path+f"hyper_parameters/{self.model_name}.json", 'w') as file:
            json.dump(self.hyperparameters, file, indent=4)
        print("Hyperparameters saved")
        
        with open(path+f"metrics/{self.model_name}.json", 'w') as file:
            json.dump(self.metrics, file, indent=4)
        print("Metrics saved")
        
        if plot:
            save_plot_html(path+f"plots/{self.model_name}.html", plot)
            
    def restore_model(self, new_hyperparameters,epoch,input_shape,output_shape):
        with open(f'{self.save_path}/hypertuning/hyper_parameters/'+self.model_name+'.json', 'r') as file:
            hyperparameters = json.load(file)
        self.set_hyperparameters(hyperparameters, new_hyperparameters=new_hyperparameters, input_shape=input_shape, output_shape=output_shape)
        self.restore_weights(epoch)
        return self.hyperparameters
    
    def restore_weights(self, epoch):
        self.parameters=dict()
        for l in range(1, len(self.layer_dims)):
            self.parameters["W" + str(l)] = tf.Variable(tf.zeros([self.layer_dims[l], self.layer_dims[l-1]]), dtype=tf.float32, trainable=True)
            self.parameters["b" + str(l)] = tf.Variable(tf.zeros([self.layer_dims[l],1]), dtype=tf.float32, trainable=True)
        self.checkpoint = tf.train.Checkpoint(parameters=self.parameters)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.save_path+"weights/", max_to_keep=5
        )
        if not epoch:
            if self.checkpoint_manager.latest_checkpoint:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                print(f"Model restored from {self.checkpoint_manager.latest_checkpoint}")
            else:
                print("No checkpoint found to restore from.")
        else:
            checkpoint_path = f"{self.checkpoint_manager.directory}ckpt-{epoch}"
            self.checkpoint.restore(checkpoint_path).expect_partial()
            print(f"Model restored from checkpoint for epoch {epoch}: {checkpoint_path}")
    
    def check_if_previous_metrics(self, costs, train_acc, test_acc):
        if os.path.exists(f"{self.save_path}/hypertuning/metrics/{self.model_name}.json"):
            with open(f"{self.save_path}/hypertuning/metrics/{self.model_name}.json", 'r') as file:
                old_metrics=json.load(file)
            old_metrics["cost"].append(costs)
            old_metrics["train_acc"].append(train_acc)
            old_metrics["test_acc"].append(test_acc)
            return old_metrics
        else:
            return {
                "cost":[costs], 
                "train_acc":[train_acc], 
                "test_acc":[test_acc]
            }

    
    def training(self, train, test, num_epochs):
        print(f"Training with hyperparameters :" + ";".join(f" {key}:{value[-1]}" for key, value in self.hyperparameters.items()))
        costs, train_acc, test_acc = list(), list(), list()
        
        self.checkpoint = tf.train.Checkpoint(parameters=self.parameters)
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.save_path+"weights/", max_to_keep=5
        )

        lr, minibatch_size, l2_lambda = self.get_hyperparameters()
        
        optimizer = tf.keras.optimizers.Adam(lr)
        
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
                if epoch%1==0 or epoch==num_epochs-1:
                    epoch_total_loss /= m
                    for (minibatch_X, minibatch_Y) in test_minibatches:
                        ZL = self.forward(tf.transpose(minibatch_X))
                        test_accuracy.update_state(minibatch_Y, tf.transpose(ZL))

                    print(f"Epoch {epoch}: cost: {epoch_total_loss}; train_acc: {train_accuracy.result().numpy()}; test_acc: {test_accuracy.result().numpy()}")
                    
                    costs.append(epoch_total_loss.numpy().astype(float))
                    train_acc.append(train_accuracy.result().numpy().astype(float))
                    test_acc.append(test_accuracy.result().numpy().astype(float))
                    test_accuracy.reset_state()
                    self.checkpoint_manager.save(checkpoint_number=epoch)
        except KeyboardInterrupt:
            print("Keyboard pressed")
        self.metrics=self.check_if_previous_metrics(costs, train_acc, test_acc)
        return self.metrics