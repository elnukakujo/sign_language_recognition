import pandas as pd
import numpy as np
from display import display_img, display_metrics
from preprocess import preprocess
from model import sign_classifier

x_train, y_train, x_test, y_test = preprocess(load=True)

model = sign_classifier()
parameters, costs, train_acc, test_acc = model.training(x_train, y_train,x_test,y_test, learning_rate = 0.00001, num_epochs = 30, minibatch_size = 128)
display_metrics(costs, train_acc, test_acc)