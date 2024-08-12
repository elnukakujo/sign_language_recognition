import pandas as pd
import numpy as np
import tensorflow as tf

def save_weights(**kwargs):
    for key, value in kwargs.items():
        path=f"data/processed/{key}.npy"
        np.save(path,value.numpy())

def load_weights(path):
    return tf.convert_to_tensor(np.load(path))

def load_csv():
    return pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")

def split_tensor(data):
    return (tf.convert_to_tensor(data.loc[:,'pixel1':'pixel784']),
            tf.convert_to_tensor(data.loc[:,'label']))
    
def image_normalization(image):
    image = tf.cast(image, tf.float32) / 255.0
    return tf.transpose(image)

def one_hot_encoder(labels):
    num_classes = tf.unique(labels).y.shape[0]
    one_hot=tf.one_hot(labels,depth=num_classes)
    return tf.transpose(one_hot)

def preprocess(load=False):
    if load:
        return (load_weights(f'data/processed/x_train.npy'), load_weights(f'data/processed/y_train.npy'),
                load_weights(f'data/processed/x_test.npy'), load_weights(f'data/processed/y_test.npy'))
    train, test = load_csv()
    x_train, y_train = split_tensor(train)
    x_test, y_test = split_tensor(test)
    x_train, x_test = image_normalization(x_train),image_normalization(x_test)
    y_train, y_test = one_hot_encoder(y_train), one_hot_encoder(y_test)
    save_weights(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    
    return x_train, y_train, x_test, y_test