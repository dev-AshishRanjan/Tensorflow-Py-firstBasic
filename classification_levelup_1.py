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

#checking my theory
columns=["SepalLength","SepalWidth","PetalLength","PetalWidth"]
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
print("Predicted class is : "+ str(probab_class) + "\n" + "With probability of : " + str(probability*100))