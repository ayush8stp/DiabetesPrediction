#importing necessary libraries
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#function for finding the predicted output for input values
def find(l):
    pred2 = clf.predict([l])
    for i in pred2:
        if(i==1):
            print("Diabetes Positive")
        else:
            print("Diabetes Negative")

#picking up the data from csv files
filename = open("diabetes.csv",'r')
dframe = pd.read_csv(filename)
#dividing the data in two parts the depended values(x) and independent value(y)
x = dframe.drop('Outcome',axis=1)
y = dframe['Outcome']

#this is done to divide the data for training and testing purpose
Xtrain ,Xtest ,Ytrain ,Ytest = train_test_split(x ,y)

#creating a classifier of decision tree
clf = tree.DecisionTreeClassifier()

#fitting the data to classifier
clf.fit(Xtrain ,Ytrain)

#predicting the result for test values
pred = clf.predict(Xtest)

#printing the accuracy of model
print("The accuracy of model is:{} ".format(accuracy_score(Ytest ,pred)))

#asking the details for user
print("Enter the details in a series of Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome  and if you don't know any value use 0 there")
l=list(map(int,input().split()))

#getting the predicted output for given value
find(l)

#plotting the graph for the values of training and testing
plt.scatter(Xtrain['Insulin'] ,Ytrain ,color='g' ,label ='Training Data')
plt.scatter(Xtest['Insulin'] ,Ytest ,color='r' ,label ='Testing Data')
plt.title('Scatter plot for dibetes patients')
plt.xlabel('Glucose quantity')
plt.ylabel('Dibetes result')
plt.legend(loc=1)
plt.plot(Xtest ,pred ,'bo',linewidth = 2,label='')
plt.show()

