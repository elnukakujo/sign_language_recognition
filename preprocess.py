import pandas as pd
import numpy as np
import tensorflow as tf

def save_dataset(**kwargs):
    for key, value in kwargs.items():
        path=f"data/tf_dataset/{key}"
        tf.data.Dataset.save(value,path)

def load_dataset(**kwargs):
    sets=list()
    for value in kwargs.values():
        path=f"data/tf_dataset/{value}"
        sets.append(tf.data.Dataset.load(path))
    return sets

def load_csv():
    return pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")

def split_tensor(data):
    return (tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(data.loc[:,'pixel1':'pixel784'], dtype=tf.float32))),
            tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(data.loc[:,'label'], dtype=tf.int32))))
    
def image_normalization(image):
    image = image / 255
    return image

def one_hot_encoder(labels):
    return tf.one_hot(labels,depth=24)

def preprocess(load=False):
    if load:
        return load_dataset(x_train="x_train", y_train="y_train", x_test="x_test", y_test="y_test")
    # Load the data from csv
    train, test = load_csv()
    
    #Split the labels and images, pass them into tensorflow datasets
    (x_train, y_train), (x_test, y_test) = split_tensor(train), split_tensor(test)

    #Normalize the images 0 to 1
    x_train, x_test = x_train.map(image_normalization),x_test.map(image_normalization)
    
    #One hot encode labels
    y_train, y_test = y_train.map(one_hot_encoder), y_test.map(one_hot_encoder)
    save_dataset(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    
    return x_train, y_train, x_test, y_test