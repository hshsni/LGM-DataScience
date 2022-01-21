import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

iris=load_iris()
#print(iris.keys())
#print(iris.DESCR)

#Seperating features into individual lists, T stands for transpose
features=iris.data.T
sepal_len=features[0]
sepal_width=features[1]
petal_len=features[2]
petal_width=features[3]

'''
#Plots a graph of sepal length versus sepal width for the target classes
plt.scatter(sepal_len,sepal_width,c=iris.target)
plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()
plt.show()
'''

#Training the model by splitting the data and targets using the train_test_split func
#Add parameter random_state=0 so that everytime it is split the same way, otherwise each time it will be completely random
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=0)

#Making the model using k nearest neighbours, other is logistic regression method and decision tree method
model=KNeighborsClassifier(n_neighbors=1)

#Training the model
model.fit(X_train,y_train)

#Checking the performance score of the model based on the test data
score=model.score(X_test,y_test)
print(score)

#Checking the class of the new data entered
sample=np.array([[5,3,4,4]])
prediction=model.predict(sample)
print(prediction)