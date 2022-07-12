import tensorflow as tf
import pandas as pd
from tensorflow._api.v2 import data, feature_column



# dataset
CSV_COLUMN_NAMES=["SepalLength","SepalWidth","PetalLength","PetalWidth","Species"]
SPECIES=["Setosa","Versicolor","Virginica"]

train_path=tf.keras.utils.get_file("iris_training.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv")
test_path=tf.keras.utils.get_file("iris_test.csv","https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv")

train=pd.read_csv(train_path,names=CSV_COLUMN_NAMES,header=0)
test=pd.read_csv(test_path,names=CSV_COLUMN_NAMES,header=0)

#print(train.head())

train_y=train.pop("Species")
test_y=test.pop("Species")

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
for key in train.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))

#print(my_feature_columns)

#building the model with DNN
classifier=tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[30,10],
    n_classes=3
    )

#training
classifier.train(input_fn=lambda: input_fn(train, train_y,training=True), steps=5000)

#evaluation
eval_result=classifier.evaluate(input_fn=lambda: input_fn(test, test_y,training=False))

print("\n Test set accuracy : {accuracy:0.3f}\n".format(**eval_result))



#prediction
p=test.loc[0]["SepalLength"]
print(p)


columns=["SepalLength","SepalWidth","PetalLength","PetalWidth"]

'''

for data in columns:
    test.loc[0][str(data)]= input(data +":")
print(test.loc[0])
prediction=classifier.predict(input_fn=lambda: input_fn(test, test_y,training=False))
prediction_list=list(prediction)
predictions=prediction_list[0]

print(predictions)
print("\n\n")
class_id = predictions["class_ids"][0]
probability=predictions["probabilities"][class_id]
probab_class=SPECIES[class_id]
print("Predicted class is : "+ str(probab_class) + "\n" + "With probability of : " + str(probability*100))   '''

## prediction using tkinter
from tkinter import *
root=Tk()
root.title("Iris flower class predictor")
#root.geometry("400x400")

for i in range(4):
    label=Label(root,text=columns[i] + "  :", font=("helvetica",12)).grid(row=i,column=0,pady=10,padx=5)

#predict function
def predict():
    global columns, SPECIES
    global e1 , e2 , e3 , e4
    

    test.loc[0]["SepalLength"]= str(e1.get())
    test.loc[0]["SepalWidth"]= str(e2.get())
    test.loc[0]["PetalLength"]= str(e3.get())
    test.loc[0]["PetalWidth"]= str(e4.get())

    prediction=classifier.predict(input_fn=lambda: input_fn(test, test_y,training=False))
    prediction_list=list(prediction)
    predictions=prediction_list[0]

    class_id = predictions["class_ids"][0]
    probability=predictions["probabilities"][class_id]
    probab_class=SPECIES[class_id]
    print("\n  Predicted class is : "+ str(probab_class) + "\n" + "With probability of : " + str(probability*100))
    label=Label(root,text="\n Predicted class is : "+ str(probab_class) + "\n" + "With probability of : " + str(probability*100) +" %" + "\n", fg="blue",font=("helvetica",10))
    label.grid(row=5,column=0,columnspan=2)
    label3=Label(root,text="We classify your data in the 3 following IRIS Flower Class : \n * Sentosa \n * Versicolor \n * Virginica  ")
    label3.grid(row=6,column=0,columnspan=2)
    label2=Label(root,text="\n\n App created with ML\n", fg="red",font=("chiller", 20))
    label2.grid(row=7,column=0,columnspan=2)

    e1.delete(0,END)
    e2.delete(0,END)
    e3.delete(0,END)
    e4.delete(0,END)

#creating entry box
e1=Entry(root, width=35)
e1.grid(row=0,column=1)

e2=Entry(root, width=35)
e2.grid(row=1,column=1)

e3=Entry(root, width=35)
e3.grid(row=2,column=1)

e4=Entry(root, width=35)
e4.grid(row=3,column=1)

Button=Button(root,text="Submit",command=predict,bd=5,bg="#D3D3D3",fg="red",padx=20,pady=5).grid(row=4,column=0,columnspan=2,pady=10)
root.mainloop()