#!/usr/bin/python

import random
import numpy as np
import matplotlib.pyplot as plt
import pickle
from outlier_cleaner import outlierCleaner
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score




def outlierRemovalRegression(features,labels):
    
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42)
    
    plt.scatter(labels_train,features_train)
    plt.xlabel("POI")
    plt.ylabel("Salary")
    plt.show()
    
    ### fill in a regression here!  Name the regression object reg so that
    ### the plotting code below works, and you can see what your regression looks like
    
    reg_model = LinearRegression()
    reg_model = reg_model.fit(features_train,labels_train)
    print("Coefficent/Slope {}".format(reg_model.coef_))
    pred = reg_model.predict(features_test)
    print("Accuracy(R2 Score) {}".format(r2_score(labels_test,pred)))
    
    
    

#     try:
#         plt.plot(ages, reg.predict(ages), color="blue")
#     except NameError:
#         pass
#     plt.scatter(ages, net_worths)
#     plt.show()


### identify and remove the most outlier-y points
    cleaned_data = []
    try:
        predictions = reg_model.predict(features_train)
        cleaned_data = outlierCleaner( predictions, features_train, labels_train )
    except NameError:
        print "your regression object doesn't exist, or isn't name reg"
        print "can't make predictions to use in identifying outliers"



    



    ### only run this code if cleaned_data is returning data
    if len(cleaned_data) > 0:
        features_train, labels_train, errors = zip(*cleaned_data)
        features_train = np.reshape( np.array(features_train), (len(features_train), 1))
        labels_train = np.reshape( np.array(labels_train), (len(labels_train), 1))
    
#         ### refit your cleaned data!
#         try:
#             reg.fit(ages, net_worths)
#             plt.plot(ages, reg.predict(ages), color="blue")
#         except NameError:
#             print "you don't seem to have regression imported/created,"
#             print "   or else your regression object isn't named reg"
#             print "   either way, only draw the scatter plot of the cleaned data"
        plt.scatter(labels_train,features_train)
        plt.xlabel("POI")
        plt.ylabel("Salary")
        plt.show()
        
    else:
        print "outlierCleaner() is returning an empty list, no refitting to be done"
        
#     print("New Slope is {}".format(reg.coef_))
#     print("New Score is {}".format(r2_score(net_worths_test,reg.predict(ages_test))))

