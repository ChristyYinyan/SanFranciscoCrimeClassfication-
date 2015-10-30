__author__ = 'easycui'
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import random

def loadData(filename):
    return pd.read_csv(filename, parse_dates=['Dates'])


def main(trainfile, testfile):
    trainRaw = loadData(trainfile)
    preTrain = _preprocessing(trainRaw)
    # perTest=_preprocessing(testRaw)
    featureAnalysis(preTrain,'day')
    feature = feature_selection()
    showXYProb(preTrain)
    cross_Validation(preTrain,feature)


def _preprocessing(data):
    encoder = preprocessing.LabelEncoder()
    crime = encoder.fit_transform(data.Category)

    # days = pd.get_dummies(data.DayOfWeek)
    # district = pd.get_dummies(data.PdDistrict)
    # hour = pd.get_dummies(data.Dates.dt.hour)
    # proData = pd.concat([hour, days, district], axis=1)
    proData = pd.DataFrame()
    # print(data['DayOfWeek'])
    print type(data['DayOfWeek']), type(data.DayOfWeek)
    proData['week'] = data.DayOfWeek
    proData['w']=encoder.fit_transform(data.DayOfWeek)
    proData['day'] = data.Dates.dt.day
    proData['year'] = data.Dates.dt.year
    proData['hour'] = data.Dates.dt.hour
    proData['PdDistrict'] =data.PdDistrict
    proData['Dis']=encoder.fit_transform(data.PdDistrict)
    proData['crime'] = crime
    proData['X'] = np.array(data.X)
    proData['Y'] = np.array(data.Y)

    return proData


def featureAnalysis(data, feature):
    """

    :param data:
    :param feature:  feature can be 'day','week','year','hour','PdDistrict'
    :return:
    """
    target = np.array(data['crime'])
    classes = len(set(target))

    featureData = list((data[feature]))
    featureData.sort()
    diffFeature = len(set(featureData))
    index = 0
    dictFeature = dict()
    for i in range(len(featureData)):
        if featureData[i] not in dictFeature:
            dictFeature[featureData[i]] = index
            index += 1
    targetCount = np.zeros((classes, diffFeature))
    XAxis = []

    for i in range(target.shape[0]):
        targetCount[target[i], dictFeature[featureData[i]]] += 1
    items = dictFeature.items()
    items.sort(key=lambda items: items[1])
    X = [x[1] for x in items]
    X1 = [x[0] for x in items]
    print X, X1
    plt.figure(figsize=(15, 5))
    plt.xticks(X, X1)
    for i in range(classes):
        Y = targetCount[i]
        plt.plot(X, Y)
    plt.show()


def showXYProb(data):
    target = np.array(data['crime'])
    classes = len(set(target))
    # discrete the X,Y
    X = np.array(data.X)
    origX = X.copy()
    maxX = X.max()
    minX = X.min()
    while maxX > -122.2:
        index = np.argmax(X)
        X = np.append(X[:index], X[index + 1:])
        maxX = X.max()
    Y = np.array(data.Y)
    origY = Y.copy()
    maxY = Y.max()
    while maxY > 45:
        index = np.argmax(Y)
        Y = np.append(Y[:index], Y[index + 1:])
        maxY = Y.max()
    minY = Y.min()
    print maxX, minX, maxY, minY
    block = 200
    scaleX = (maxX - minX) / block
    scaleY = (maxY - minY) / block
    X1 = np.zeros(len(X), dtype=int)
    Y1 = np.zeros(len(X), dtype=int)
    for i in range(len(X)):
        X1[i] = int((origX[i] - minX) / scaleX)
        Y1[i] = int((origY[i] - minY) / scaleY)
    prob = np.zeros((classes, block, block))
    print(np.array(X1))
    for i in range(len(X1)):
        if Y1[i] >= block or X1[i] >= block:
            continue
        prob[target[i], X1[i], Y1[i]] += 1
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X = np.zeros((block, block))
    Y = np.zeros((block, block))
    for i in range(block):
        for j in range(block):
            X[i, j] = i
            Y[i, j] = j
    colors = ['b', 'g', 'r', 'c', 'm']
    for i in range(classes):
        Z = prob[i, :, :]
        ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10, color=(random.random(),random.random(),random.random()))

    plt.show()


def feature_selection():
    features = ['day', 'year', 'X', 'Y', 'Dis']
    return features


def cross_Validation(data, feature, model=LogisticRegression):
    print(data)
    training, validation = train_test_split(data, train_size=0.8)

    m = model()
    m.fit(training[feature], training['crime'])
    predicted_prob = np.array(m.predict_proba(validation[feature]))
    logLoss = log_loss(validation['crime'], predicted_prob)

    print "log loss is ", logLoss

    error = 0
    return error


if __name__ == "__main__":
    trainfile = './train.csv'
    testfile = './test.csv'
    main(trainfile, testfile)
