import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt


#dataset
fashion_mnist= keras.datasets.fashion_mnist
(train_images,train_labels) , (test_images,test_labels) = fashion_mnist.load_data()

#checking data
#test_images.shape
train_images[0,23,23]
train_labels[:10]

class_names=["T-shirt", "Trousers","Pullover","Dress","Coat","Sandal","Shirt","Sneakers","Bag","Ankle boot"]

'''
#finally let's look at some images of dataset
plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()   '''

#data processing
train_images=train_images/255.0
test_images=test_images/255.0

#building the Model

model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128,activation="relu") ,
    keras.layers.Dense(10,activation="softmax")
])

#compile the model

model.compile(optimizer="adam",
loss="sparse_categorical_crossentropy",
metrics=["accuracy"])

#training and evaluating
model.fit(train_images,train_labels,epochs=10)

test_loss,test_acc= model.evaluate(test_images,test_labels, verbose=1)
print("Test accuracy : ", test_acc)

#making predictions
predictions=model.predict(test_images)

print(predictions[0])
print("\n\n","*"*30)
p=int(input("Enter a number: "))
print(predictions[p])
print(np.argmax(predictions[p]))
print("Prediction is : ",class_names[np.argmax(predictions[p])])
print("Actual data shows : ",class_names[test_labels[p]])
#finally let's look at some images of dataset
plt.figure()
plt.imshow(test_images[p])
plt.colorbar()
plt.grid(False)
plt.show()
