import pandas as pd
import numpy as np
from display import display_img, display_metrics
from preprocess import preprocess
from model import sign_classifier

x_train, y_train, x_test, y_test = preprocess(load=True)

model = sign_classifier()
for _ in range(0, 25):
    hidden_nodes = np.sort(np.random.randint(79,100,size=2))[::-1].tolist()
    r=np.random.rand()*-2
    learning_rate = (6-r)**(-5)
    minibatch_size=2**9
    parameters, costs, train_acc, test_acc = model.training(x_train, y_train, x_test, y_test, hidden_nodes, learning_rate, minibatch_size, num_epochs = 50)
    title=f"Evolution of Cost, and Training/Test Accuracy over 30 epochs for learning rate={learning_rate}, hidden nodes={hidden_nodes} and mini batch size={minibatch_size}"
    display_metrics(costs, train_acc, test_acc, title)
    print(f"Training finished: cost:{costs[-1].numpy()}; train_acc:{train_acc[-1].numpy()}; test_acc:{test_acc[-1].numpy()}")