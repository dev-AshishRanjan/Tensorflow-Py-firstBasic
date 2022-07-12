from __future__ import absolute_import,division,print_function,unicode_literals

import tensorflow as tf

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from six.moves import urllib

#import tensorflow.compat.v2.feature_column as fc

#loading Data
dftrain= pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/train.csv")
dfeval=pd.read_csv("https://storage.googleapis.com/tf-datasets/titanic/eval.csv")

y_train=dftrain.pop("survived")
y_eval=dfeval.pop("survived")

###plotting some data
#print(dftrain.loc[0])
#dftrain.age.hist(bins=20)
#dftrain.sex.value_counts().plot(kind="barh")
#pd.concat([dftrain,y_train], axis=1).groupby("sex").survived.mean().plot(kind="barh").set_xlabel(" % survive")
#plt.grid(False)
#plt.show()

#creating proper dataset
CATEGORICAL_COLUMN=["sex", "n_siblings_spouses","parch","class","deck","embark_town","alone"]
NUMERIC_COLUMN=["age","fare"]

feature_columns=[]

for feature_name in CATEGORICAL_COLUMN:
    vocabulary=dftrain[feature_name].unique()
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name,vocabulary))

for feature_name in NUMERIC_COLUMN:
    feature_columns.append(tf.feature_column.numeric_column(feature_name,dtype=tf.float32))

#print(feature_columns)

#creating input function
def make_input_fn(data_df,label_df,num_epochs=10,shuffle=True,batch_size=32):
    def input_function():
        ds=tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))
        if shuffle:
            ds=ds.shuffle(1000)
        ds=ds.batch(batch_size).repeat(num_epochs)
        return ds
    return input_function

#creating input values
train_input_fn=make_input_fn(dftrain,y_train)
eval_input_fn=make_input_fn(dfeval,y_eval,num_epochs=1,shuffle=False)

#creating the model

linear_est=tf.estimator.LinearClassifier(feature_columns=feature_columns)

#training the model
linear_est.train(train_input_fn)
result=linear_est.evaluate(eval_input_fn)

print(result["accuracy"])

# making specific predictions
result2=list(linear_est.predict(eval_input_fn))
s=result2[0]["probabilities"][1]   #probab for survival
print(dfeval.loc[0])
print(y_eval[0])
print("\n\n\n")
print("*"*148)
print(f" percentage of survival of the person is : {s*100} ")
print(f"actual survival data is: {y_eval[0]} \t 0 = did not survive")
print("*"*148)
print("\n\n\n")

#making random prediction
n=int(input("enter the index number to see"))

result3=list(linear_est.predict(eval_input_fn))
s=result3[n]["probabilities"][1]   #probab for survival
print(dfeval.loc[n])
print(y_eval[n])
#print("\n\n\n")
print("+"*148)
print(f" percentage of survival of the person is : {s*100} ")
print(f"actual survival data is: {y_eval[n]} \t 0 = did not survive")
print("+"*148)
#print(f" model accuracy :{result ["accuracy"]} " )
print("\n\n\n")