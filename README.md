# Sign Language Recognition Neural Network using Tensorflow, OpenCV and MediaPipe
The data used is the Sign Language MNIST available on [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist?resource=download).

## Preprocess
The images come as 28 by 28 matrices of black and white pixels. I tried multiple method to raise the results of the model.
1. I augmented the size of the images to 64x64 using OpenCV.
2. Using Mediapipe to detect hands and dress a skeleton to remove all the noise of the pictures. MediaPipe requires RGB pixels so I reproduced 3 times the black and white values to respect the template. Sadly, Mediapipe couldn't detect anything, so I dropped it.
3. Using OpenCV once again, I used the Canny edge detection to try to remove some of the noise coming from the variation in the pixels. Still available to use, but the model doesn't perform well on them.

I normalized the pixels between 0-1, one hot encoded the 24classes passed the values in tf.data.Dataset.

## Model
The model is made without keras so most of the low level operations are defined (forward propagation operations, iterating through the epochs, saving the parameters and hyper parameters). The optimizer, the L2 regularization, the computation of the gradients and metrics are done by tensorflow functions. I limited myself to not use CNN and to work with the tools I know well for now to test their limits.

## Hypertuning
I mainly focused on the learning rate, the number of hidden layers and their nodes, but I also worked on the mini batch size and the l2 lambda. I sampled them by for some in a uniform sampling, other through a log scale random sampling. I tried two ways:
1. The caviar method: sampling all the hyperparameters, and trying 10 times, looking at the plotly result graph and resampling a gain with a better scale. I did this method at first when the computations were fast
2. The panda method: sampling once, training the model by keeping an eye on him and changing hyper parameters when needed. I finished with this method.

## Results
My best result with 81% train accuracy, a high variance of 17% and a cost of 0.67.
![image](https://raw.githubusercontent.com/elnukakujo/sign_language_recognition/main/data/plots/20240816_172458.png)
