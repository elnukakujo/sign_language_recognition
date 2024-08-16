import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#To remove the useless messages like 
# 2024-08-14 21:33:12.845852: I tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: 
# OUT_OF_RANGE: End of sequence

import pandas as pd
import numpy as np
from display import display_img, display_metrics, compare_metric, save_plot_html
from preprocess import preprocess
from model import sign_classifier
from datetime import datetime

def generate_hyperparameters():
    r = np.random.uniform(np.log10(4e-6),np.log10(5e-6))  #np.random.uniform(np.log10(7e-4),np.log10(8e-6))
    lr= 5e-7
    hidden_nodes = [85,84] #np.sort(np.random.randint(70,100,size=4))[::-1].tolist()
    minibatch_size=2**9
    l2_lambda = 0
    return {
        "lr":[lr],
        "hidden_nodes":[hidden_nodes],
        "minibatch_size":[minibatch_size],
        "l2_lambda":[l2_lambda]
    }

def train_model(train, test, model_name=False, epoch=False):
    if model_name:
        model = sign_classifier(model_name)
        new_hyperparameters = generate_hyperparameters()
        for image, label in train.take(1):
            input_shape = image.numpy().shape[0]
            output_shape = label.numpy().shape[0]
            hyperparameters = model.restore_model(new_hyperparameters, epoch, input_shape, output_shape)
    else:
        model_name = datetime.now().strftime('%Y%m%d_%H%M%S')
        model = sign_classifier(model_name)
        hyperparameters=generate_hyperparameters()
        for image, label in train.take(1):
            input_shape = image.numpy().shape[0]
            output_shape = label.numpy().shape[0]
            model.set_hyperparameters(hyperparameters, input_shape, output_shape)
        model.initialize_parameters()
        
    metrics = model.training(train, test, num_epochs = 1000)
    
    title=f"Evolution of Cost, and Training/Test Accuracy for:" + ";".join(f" {key}:{value[-1]}" for key, value in hyperparameters.items())
    plot = display_metrics(metrics, title, hyperparameters)
    print(f"Training finished:" + ";".join(f" {key}:{value[-1][-1]}" for key, value in metrics.items()))
    
    model.save_results(plot)
    
    return hyperparameters, metrics, plot

train, test = preprocess(load=True, edge_detect=False, size=(64,64))

show_img=False
if show_img:
    for image, label in train.take(1):
        display_img(image.numpy().reshape(64,64), np.argmax(label))

hyperparameters, metrics, plot = train_model(train,test,model_name="20240816_172458", epoch=475)#model_name="20240816_172458", epoch=368