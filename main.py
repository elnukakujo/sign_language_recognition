import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#To remove the useless messages like 
# 2024-08-14 21:33:12.845852: I tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: 
# OUT_OF_RANGE: End of sequence

import pandas as pd
import numpy as np
from display import display_img, display_metrics, compare_metric
from preprocess import preprocess
from model import sign_classifier

train, test = preprocess(load=True, edge_detect=False, size=(64,64))
show_img=False
if show_img:
    for image, label in train.take(1):
        display_img(image.numpy().reshape(64,64), np.argmax(label))

model = sign_classifier()
hyper_parameters = []
train_accs=[]
test_accs=[]
try: 
    for training in range(0, 10):
        hidden_nodes = np.sort(np.random.randint(65,75,size=2))[::-1].tolist()
        r = np.random.rand()
        learning_rate = 10**(-5-r)
        minibatch_size=2**7
        r = np.random.rand()*2
        l2_lambda = r
        print(f"Training {training}")
        parameters, costs, train_acc, test_acc = model.training(train, test, hidden_nodes, learning_rate, minibatch_size, num_epochs = 150, l2_lambda=l2_lambda)
        hyper_parameters.append({
            "learning_rate":learning_rate,
            "hidden_nodes":hidden_nodes,
            "minibatch_size":minibatch_size,
            "l2_lambda":l2_lambda
        })
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        """title=f"Evolution of Cost, and Training/Test Accuracy for learning rate={learning_rate}, hidden nodes={hidden_nodes}, mini batch size={minibatch_size} and l2_lambda={l2_lambda}"
        display_metrics(costs, train_acc, test_acc, title)"""
        print(f"Training finished: cost:{costs[-1].numpy()}; train_acc:{train_acc[-1].numpy()}; test_acc:{test_acc[-1].numpy()}")
except IndexError:
    print("IndexError: Results empty for this training")

if len(train_accs)!=0:
    title="Comparison of the train_accuracy along the trainings"
    compare_metric(train_accs,title,hyper_parameters)
    
if len(test_accs)!=0:
    title="Comparison of the test_accuracy along the trainings"
    compare_metric(test_accs,title,hyper_parameters)