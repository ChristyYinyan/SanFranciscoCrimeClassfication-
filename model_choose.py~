__author__ = 'yinyan'
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
import numpy as np

train_data=pd.read_csv("train_data_output.csv")
feature1=['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday',
          'Tuesday','Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE',
          'MISSION','NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL',
          'TENDERLOIN']
add_feature=[str(x) for x in range(24)]
feature2=feature1+add_feature
feature3 = feature1 + ['X', 'Y']
#526829 used for training,  351220 for validation
training, validation = train_test_split(train_data, train_size=.60)
def model():
    model = LogisticRegression()
    features=feature3
    #fit Naive Bayes classifier according to X, y, X are the features, y is crime
    model.fit(training[features], training['crime'])
    #classfication of vectors X
    # predicted= np.array(model.predict(validation[features]))
    print "the accuracy of this model", model.score(validation[features], validation['crime'])

    """predict_proba: return the probability estimates for the test vectors"""
    predicted_prob = np.array(model.predict_proba(validation[features]))
    print(predicted_prob[1])
    # predicted_prob_csv=pd.DataFrame(predicted_prob, columns=le_crime.classes_)
    # BernoulliNB_result=predicted_prob_csv.to_csv('testResult_Model1.csv', index=True, index_label='Id')
    log_loss_value1=log_loss(validation['crime'], predicted_prob)
    print "log_loss for this model", log_loss_value1








if __name__=='__main__':
    model()


