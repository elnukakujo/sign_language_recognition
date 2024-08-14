import pandas as pd
import numpy as np
from display import display_img, display_metrics
from preprocess import preprocess
from model import sign_classifier

x_train, y_train, x_test, y_test = preprocess(load=False)
for image in x_train.take(1):
    display_img(image.numpy().reshape(28,28))

"""model = sign_classifier()
for _ in range(0, 25):
    hidden_nodes = np.sort(np.random.randint(80,100,size=4))[::-1].tolist()
    r=np.random.rand()
    learning_rate = 10**(-5-r)
    minibatch_size=2**6
    l2_lambda = 1
    parameters, costs, train_acc, test_acc = model.training(x_train, y_train, x_test, y_test, hidden_nodes, learning_rate, minibatch_size, num_epochs = 50, l2_lambda=l2_lambda)
    title=f"Evolution of Cost, and Training/Test Accuracy over 50 epochs for learning rate={learning_rate}, hidden nodes={hidden_nodes}, mini batch size={minibatch_size} and l2_lambda={l2_lambda}"
    display_metrics(costs, train_acc, test_acc, title)
    print(f"Training finished: cost:{costs[-1].numpy()}; train_acc:{train_acc[-1].numpy()}; test_acc:{test_acc[-1].numpy()}")"""