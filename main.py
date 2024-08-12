import pandas as pd
import numpy as np
from display import display_img
from preprocess import preprocess

x_train, y_train, x_test, y_test = preprocess(load=False)
print(x_train)