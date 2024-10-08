import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#To remove the useless messages like 
# 2024-08-14 21:33:12.845852: I tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: 
# OUT_OF_RANGE: End of sequence

import pandas as pd
import numpy as np
from display import display_img, display_metrics, compare_metric, save_plots_html
from preprocess import preprocess
from model import sign_classifier
from datetime import datetime
import json

def save_results(hyperparameters, train_plot, test_plot):
    filename = datetime.now().strftime('%Y%m%d_%H%M%S')
    path=f"hypertuning/caviar/hyper_parameters/{filename}.json"
    with open(path, 'w') as file:
        json.dump(hyperparameters, file, indent=4)
    print("Hyperparameters saved")
    save_plots_html("hypertuning/caviar/", filename, train_plot, test_plot)

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
    for training in range(0, 3):
        hidden_nodes = np.sort(np.random.randint(70,100,size=4))[::-1].tolist() # np.sort(np.random.randint(65,75,size=2))[::-1].tolist()
        r = np.random.uniform(np.log10(6e-7),np.log10(4e-6))
        learning_rate = 10**(r)
        minibatch_size=2**6
        l2_lambda = 0
        print(f"Training {training}")
        parameters, costs, train_acc, test_acc = model.training(train, test, hidden_nodes, learning_rate, minibatch_size, num_epochs = 200, l2_lambda=l2_lambda)
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
    
if len(train_accs)!=0 and len(test_accs)!=0:
    title="Comparison of the train_accuracy along the trainings"
    train_plot = compare_metric(train_accs,title,hyper_parameters)
    
    title="Comparison of the test_accuracy along the trainings"
    test_plot = compare_metric(test_accs,title,hyper_parameters)

    save_results(hyper_parameters,train_plot,test_plot)