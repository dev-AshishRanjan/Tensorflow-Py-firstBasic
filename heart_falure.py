import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow_estimator.python.estimator.estimator import Estimator

#load data
dftrain= pd.read_csv("heart_failure_clinical_records_dataset.csv")
dftest= pd.read_csv("heart_failure_clinical_records_dataset2.csv")
#print(dftrain)
y_train=dftrain.pop("DEATH_EVENT")
y_test=dftest.pop("DEATH_EVENT")
#print(y_train)

NUMERIC_COLUMN=["age","anaemia","creatinine_phosphokinase","diabetes","ejection_fraction","high_blood_pressure","platelets","serum_creatinine","serum_sodium","sex","smoking","time"]
my_feature_columns=[]

#creating input function
def input_fn(features,labels,training=True,batch_size=256):
    #convert the input to a dataset
    dataset=tf.data.Dataset.from_tensor_slices((dict(features),labels))
    #shuffle and repeat if you are in training mode
    if training:
        dataset=dataset.shuffle(1000).repeat()

    return dataset.batch(batch_size)

#feature column
my_feature_columns=[]
for key in dftrain.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

#print(my_feature_columns)

#building the model with DNN
classifier=tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30,10],
    n_classes=2
    )

#training
classifier.train(input_fn=lambda: input_fn(dftrain, y_train,training=True), steps=5000)

#evaluation
eval_result=classifier.evaluate(input_fn=lambda: input_fn(dftest, y_test,training=False))

print("\n Test set accuracy : {accuracy:0.3f}\n".format(**eval_result))



#prediction
#p=dftest.loc[0]["SepalLength"]
#print(p)

#checking my theory
#columns=["SepalLength","SepalWidth","PetalLength","PetalWidth"]

'''
for data in NUMERIC_COLUMN:
    dftest.loc[0][str(data)]= input(data +":")
print(dftest.loc[0])
prediction=classifier.predict(input_fn=lambda: input_fn(dftrain, y_train,training=False))
prediction_list=list(prediction)
predictions=prediction_list[0]

survival=["deth","survive"]
print(predictions)
print("\n\n")
class_id = predictions["class_ids"][0]
probability=predictions["probabilities"][class_id]
probab_class=survival[class_id]
print("Predicted class is : "+ str(probab_class) + "\n" + "With probability of : " + str(probability*100))      

p=dftest.loc[0]
print(p)
p["age"]=50     '''

def pred():

    pred=np.zeros([1,13])
    dictp={}
    pred[0]= [64,1,147,1,30,0,150000,1.2,130,0,1,7,1]
    for i in range(len(NUMERIC_COLUMN)):
        dictp[NUMERIC_COLUMN[i]]=pred[0][i]
    p=tf.data.Dataset.from_tensor_slices(dictp).batch(1)
    result=classifier.predict(p)
    return result
#p=list(result)
print(list(pred()))

