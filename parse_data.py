__author__ = 'yinyan'
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np

#Load Data with pandas, and parse the first column into datetime
train=pd.read_csv('./train.csv', parse_dates = ['Dates'])
test=pd.read_csv('./test.csv', parse_dates = ['Dates'])
###############################################################
# Date---date + timestamp
# Category--The type of crime, Larceny, etc.
# Descript--A more detailed description of the crime.
# DayOfWeek--Day of crime: Monday, Tuesday, etc.
# PdDistrict--Police department district.
# Resolution--What was the outcome, Arrest, Unfounded, None, etc.
# Address--Street address of crime.
# X and Y--GPS coordinates of crime.
##################################################################
le_crime = preprocessing.LabelEncoder()
def parse_Train_Data():
    crime = le_crime.fit_transform(train.Category)
    print(np.array(train.X))
    X=np.array(train.X)
    maxX=X.max()
    minX=X.min()

    Y=np.array(train.Y)
    origY=Y.copy()
    maxY=Y.max()
    while maxY>45:
        index=np.argmax(Y)
        Y=np.append(Y[:index],Y[index+1:])
        maxY=Y.max()
    minY=Y.min()
    print maxX,minX,maxY,minY
    scaleX=(maxX-minX)/100
    scaleY=(maxY-minY)/100

    X1=np.zeros(len(X))
    Y1=np.zeros(len(X))

    for i in range(len(X)):
        X1[i]=int((X[i]-minX)/scaleX)
        Y1[i]=int((origY[i]-minY)/scaleY)
    print(np.array(X1))
    days = pd.get_dummies(train.DayOfWeek)
    district = pd.get_dummies(train.PdDistrict)
    hour = train.Dates.dt.hour
    hour = pd.get_dummies(hour)
    train_data = pd.concat([hour, days, district], axis=1)
    train_data['crime']=crime
    train_data['X']=X1
    train_data['Y']=Y1
    print train_data
    return train_data

def parse_Test_data():
    days = pd.get_dummies(test.DayOfWeek)
    district = pd.get_dummies(test.PdDistrict)

    hour = test.Dates.dt.hour
    hour = pd.get_dummies(hour)

    test_data = pd.concat([hour, days, district], axis=1)
    #Save test data into a csv file
    #test_data.to_csv("test_data_output.csv", sep='\t', encoding='utf-8')
    return test_data
train_data=parse_Train_Data()
features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
     'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',
     'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']
features2 = [x for x in range(0,24)]
features = features + ['X','Y']
#526829 used for training,  351220 for validation
training, validation = train_test_split(train_data, train_size=.60)
ground_truth=np.array(validation['crime'])
# def model1_BernoulliNB():
#     model = BernoulliNB()
#     #fit Naive Bayes classifier according to X, y, X are the features, y is crime
#     model.fit(training[features], training['crime'])
#     #classfication of vectors X
#     # predicted= np.array(model.predict(validation[features]))
#     print "the accuracy of this model", model.score(validation[features], validation['crime'])
#
#     """predict_proba: return the probability estimates for the test vectors"""
#     predicted_prob = np.array(model.predict_proba(validation[features]))
#     predicted_prob_csv=pd.DataFrame(predicted_prob, columns=le_crime.classes_)
#     BernoulliNB_result=predicted_prob_csv.to_csv('testResult_Model1.csv', index=True, index_label='Id')
#     log_loss_value1=log_loss(validation['crime'], predicted_prob)
#     print "log_loss for BernolliNB model", log_loss_value1
#     #log_loss for BernolliNB model 2.58282473994

def model2_Log_rregression():
    #Logistic Regression for comparison
    model = LogisticRegression(C=.01)
    model.fit(training[features], training['crime'])
    predicted_prob = np.array(model.predict_proba(validation[features]))
    print "the accuracy of this model", model.score(validation[features], validation['crime'])

    predicted_prob_csv=pd.DataFrame(predicted_prob, columns=le_crime.classes_)
    logistic_regre_result=predicted_prob_csv.to_csv('testResult_Model2.csv', index=True, index_label='Id')
    log_loss_value2=log_loss(validation['crime'], predicted_prob)
    print "log_loss for logistic regression", log_loss_value2
    #log_loss for logistic regression 2.59190795784




if __name__=='__main__':
    model2_Log_rregression()


