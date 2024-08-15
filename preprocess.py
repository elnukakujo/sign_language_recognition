import pandas as pd
import numpy as np
import tensorflow as tf
import cv2

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

def resize_img(img, size):
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)

def split_numpy(data, size):
    images = data.loc[:,"pixel1":"pixel784"].to_numpy()
    images_reshaped = images.reshape(images.shape[0],28,28).astype(np.uint8)
    if size:
        images_reshaped=np.array([resize_img(image, size) for image in images_reshaped])
    labels = data.loc[:,"label"].to_numpy()
    return (images_reshaped,
            labels)

def to_tfdataset(images, label):
    images = images.reshape(images.shape[0],64**2)
    return (tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(images, dtype=tf.float32))),
            tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(label, dtype=tf.int32))))
    
def image_normalization(image):
    image = image / 255
    return image

def one_hot_encoder(labels):
    return tf.one_hot(labels,depth=24)

def edge_detection(images):
    return np.array([cv2.Canny(img, 100, 200) for img in images])

def preprocess(load=False, edge_detect=False, size=False):
    if load:
        return load_dataset(train="train",test="test")
    # Load the data from csv
    train, test = load_csv()
    
    #Split the labels and images, pass them into tensorflow datasets
    (x_train, y_train), (x_test, y_test) = split_numpy(train, size), split_numpy(test, size)

    if edge_detect:
        x_train, x_test = edge_detection(x_train),edge_detection(x_test)
    
    #Normalize the images 0 to 1
    x_train, x_test = image_normalization(x_train),image_normalization(x_test)
    
    #Convert to tensors
    (x_train, y_train), (x_test, y_test) = to_tfdataset(x_train,y_train), to_tfdataset(x_test,y_test)
    
    #One hot encode labels
    y_train, y_test = y_train.map(one_hot_encoder), y_test.map(one_hot_encoder)
    train,test=tf.data.Dataset.zip((x_train,y_train)),tf.data.Dataset.zip((x_test,y_test))
    save_dataset(train=train, test=test)
    
    return train, test