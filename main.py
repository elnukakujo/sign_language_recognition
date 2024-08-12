import pandas as pd
import numpy as np
import plotly.express as px

test = pd.read_csv("data/test.csv")
x_test = test.loc[:,'pixel1':'pixel784'].to_numpy(np.float32).transpose()/255.0
y_test = test.loc[:,'label'].to_numpy(np.int32).reshape(1, -1)

idx=2
img = x_test[:,idx].reshape(28,28)
label = chr(y_test[0, idx] + ord('A'))
fig = px.imshow(img, color_continuous_scale='gray')
fig.update_layout(
    title=f'This is a {label.upper()}',
    xaxis=dict(showticklabels=False),
    yaxis=dict(showticklabels=False),
)
fig.show()