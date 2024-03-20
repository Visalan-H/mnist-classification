# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Import TensorFlow library as tf, or TensorFlow's Keras interface for a higher-level API.

### STEP 2:
Utilize tf.keras.datasets.mnist.load_data() to obtain both training and testing datasets.
Normalize pixel values by dividing by 255 to enhance training efficiency.
Consider implementing one-hot encoding for labels to facilitate multi-class classification.

### STEP 3:
Create a sequential model,I have added 2 Convolutional layers of 32 and 64 filters respectively and 2 max pooling layers of fliters having height 2 and width 2
And then two dense layers having 128 and 10 neurons respectively

### STEP 4:
Define the optimizer (e.g., Adam), loss function (e.g., categorical_crossentropy), and metrics (e.g., accuracy) for model compilation.

### STEP 5:
Train the model using preferred epochs and include validation data to monitor the performance over time

### STEP 6:
Predict a new image and display its predictions
## PROGRAM

### Name:Visalan H
### Register Number:212223240183
```python
import numpy as np
import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
imgg=X_train[1]
imgg.shape
plt.imshow(imgg,cmap='gray')

y_train.shape
X_train.min()
X_train.max()
X_train_scaled=X_train/255.0
X_test_scaled=X_test/255.0
X_train_scaled.min()
X_train_scaled.max()

y_train[1]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape

imgg=X_train[8473]
plt.imshow(imgg,cmap='gray')
y_train[8473]
y_train_onehot[8473]

X_train_scaled.shape
X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model=keras.Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=10,batch_size=64,
          validation_data=(X_test_scaled,y_test_onehot))
metrics = pd.DataFrame(model.history.history)
metrics.head()

print("Visalan H")
print("212223240183")
metrics[['accuracy','val_accuracy']].plot()
print("Visalan H")
print("212223240183")
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print("Visalan H")
print("212223240183")
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))

img = image.load_img('5.png')
type(img)
plt.imshow(img,cmap='gray')

img = image.load_img('5.png')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_scaled.reshape(1,28,28,1)),
     axis=1)

print("Visalan H")
print("212223240183")
print(x_single_prediction)

print("Visalan H")
print("212223240183")
plt.imshow(img_28_gray_scaled.reshape(28,28),cmap='gray')

img_28_gray_inverted = 255.0-img_28_gray
img_28_gray_inverted_scaled = img_28_gray_inverted.numpy()/255.0

x_single_prediction = np.argmax(
    model.predict(img_28_gray_inverted_scaled.reshape(1,28,28,1)),
     axis=1)

print(x_single_prediction)
plt.imshow(img_28_gray_inverted_scaled.reshape(28,28),cmap='gray')
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![image](https://github.com/Visalan-H/mnist-classification/assets/152077751/25afd2f5-eff5-49b9-b8d2-630e07d31bc7)

![image](https://github.com/Visalan-H/mnist-classification/assets/152077751/a43e3ba3-40ef-4679-b40e-3d18c71348ee)

### Classification Report
![image](https://github.com/Visalan-H/mnist-classification/assets/152077751/4cbd27b2-1598-4541-b438-a5784297ccb6)
### Confusion Matrix
![image](https://github.com/Visalan-H/mnist-classification/assets/152077751/eb22f6a6-d920-4be3-9b5f-550593be3a94)
### New Sample Data Prediction
Sample Input:
![image](https://github.com/Visalan-H/mnist-classification/assets/152077751/6c1bd390-63e2-4b92-90c5-31f0a2d58e0a)

![image](https://github.com/Visalan-H/mnist-classification/assets/152077751/dee36b27-2f78-42b5-b867-987d050c80dd)

## RESULT
Thus, a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully.
