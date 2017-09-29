# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:10:04 2017

@author: Sayam Ganguly
"""

import _pickle as pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
import sys
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

def data_label_split(data):
    X = data['TITLE']
    Y = data['CATEGORY']
    return X,Y

stdout = sys.stdout
all_set = pickle.load(open("all_set.p","rb"))
testing_set = pickle.load(open("testing_set.p","rb"))

x,y = data_label_split(all_set)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,    
                             preprocessor = None,
                             ngram_range = (1, 1),
                             binary = False,
                             strip_accents='unicode')

x = vectorizer.fit_transform(x)

vectorized_train,vectorized_test,train_y,test_y = train_test_split(x, y, test_size=0.3)


classifier_set = [(MultinomialNB(),'Multinomial Naive Bayes'),
                  (LogisticRegression(),'Logistic Regression'),
                  (LinearSVC(),'Linear SVM')]
#                  (RandomForestClassifier(n_estimators=20),'Random Forrest')
#                  (AdaBoostClassifier(),'AdaBoost')]

class_names = ['Business', 'Technology', 'Entertainment', 'Medicine']

pred_results = {}
                  
for elem in classifier_set:
    model = elem[0]
    model_name = elem[1]
    print("Running",model_name)
    classifier = model.fit(vectorized_train, train_y)
    prediction = classifier.predict(vectorized_test)
    
    precision = metrics.precision_score(test_y, prediction,average=None)
    recall = metrics.recall_score(test_y, prediction,average=None)
    f1 = metrics.f1_score(test_y, prediction,average=None)
    accuracy = metrics.accuracy_score(test_y, prediction)
    confusion_matrix = metrics.confusion_matrix(test_y, prediction, labels=np.unique(y))
    report = classification_report(test_y, prediction, target_names=class_names)
    
    d = {'precision':precision,
         'recall':recall,
         'F1':f1,
         'accuracy':accuracy,
         'confusion_matrix':confusion_matrix,
         'report':report}
    pred_results[model_name] = d
    print("Done......")
    
sys.stdout = open("all_bagofwords_results.txt", 'a')
    
for model_name,result in pred_results.items():
    print ('-------'+'-'*len(model_name))
    print ('MODEL:', model_name)
    print ('-------'+'-'*len(model_name))
    
    print ('Precision = ' + str(result['precision']))
    print ('Recall = ' + str(result['recall']))
    print ('F1 = ' + str(result['F1']))
    print ('Accuracy = ' + str(result['accuracy']))
    print ('Confusion matrix =  \n' + str(result['confusion_matrix']))
    print ('\nClassification Report:\n' + str(result['report']))

sys.stdout.close()
sys.stdout = stdout
print("Completed!")