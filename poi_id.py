#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from time import time
from sklearn.metrics import precision_score,recall_score

sys.path.append("../Enron_Corpus/tools")
from feature_format import featureFormat, targetFeatureSplit
import tester as ts

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['salary','bonus'] # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Visualize the Data
def visualize():
    data = featureFormat(data_dict,features_list)
    for point in data:
        salary = point[0]
        bonus = point[1]
        plt.scatter( salary, bonus )
    plt.xlabel("salary")
    plt.ylabel("bonus")
    plt.show()

    # labels,features = targetFeatureSplit(data)
    # plt.scatter(labels,features)
    # plt.xlabel("POI")
    # plt.ylabel("Salary")
    # plt.show()
visualize()

### Task 2: REMOVE OUTLIERS

#Finding the outlier
max_salary,max_key = 0,''
for key,value in data_dict.items():
    if value['salary'] > max_salary and value['salary']<>'NaN':
        max_salary, max_key = value['salary'],key
        
print(max_salary,max_key)

#Removing the outlier
data_dict.pop('TOTAL')

#Visualize after removing the outlier
visualize()

outliers = []
for key in data_dict:
    sal = data_dict[key]['salary']
    if sal<>'NaN':
        outliers.append((key,int(sal)))

outliers_final = (sorted(outliers,key=lambda x:x[1],reverse=True)[:4])
print(outliers_final)

### Task 3: CREATE NEW FEATURE(S)

#Selecting two features from the existing data.
features_list = ['poi','from_poi_to_this_person','from_this_person_to_poi']
data = featureFormat(data_dict,features_list)

for point in data:
    from_poi = point[1]
    to_poi = point[2]
    if point[0]==1:
        plt.scatter(from_poi,to_poi,c="r")
    else:
        plt.scatter(from_poi,to_poi, c='b')

plt.xlabel("No.of emails from POI to this person")
plt.ylabel("NO.of emails from this person to POI")
plt.show()

def fraction_feature(new_feature,key,norm):
    
    for emp in data_dict.keys():
        
        if data_dict[emp][key]<>'NaN' and data_dict[emp][norm]<>'NaN':
            div = float(data_dict[emp][key])/data_dict[emp][norm]
            data_dict[emp][new_feature] = round(div,3)
        else:
            data_dict[emp][new_feature] = 0.0

fraction_feature("fraction_from_poi_email","from_poi_to_this_person","to_messages")
fraction_feature("fraction_to_poi_email","from_this_person_to_poi","from_messages")

#new features
features_list = ['poi',"fraction_from_poi_email","fraction_to_poi_email"]
data = featureFormat(data_dict,features_list)

for point in data:
    from_poi = point[1]
    to_poi = point[2]
    if point[0]==1:
        plt.scatter(from_poi,to_poi,c="r")
    else:
        plt.scatter(from_poi,to_poi, c='b')

plt.xlabel("fraction of emails this person gets from poi")
plt.ylabel("fraction of emails this person sends to poi")
plt.show()

## Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
features_list = ['poi',"fraction_from_poi_email","fraction_to_poi_email","shared_receipt_with_poi"]
data = featureFormat(data_dict,features_list)
labels,features=targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1,
                                                                             random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
t1 = time()
clf = GaussianNB()
clf = clf.fit(features_train, labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print 'Accuracy',round(accuracy,2)
print "Precision: ",round(precision_score(labels_test,pred),2)
print "Recall: ", round(recall_score(labels_test,pred),2)
print "NB algo. time",round(time()-t1, 3),'sec'

from sklearn.tree import DecisionTreeClassifier
t1 = time()
clf = DecisionTreeClassifier(min_samples_split=5)
clf = clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(pred,labels_test)
print 'Accuracy',round(accuracy,2)
print "Precision: ",round(precision_score(labels_test,pred),2)
print "Recall: ", round(recall_score(labels_test,pred),2)
print "DecisionTree Clf algo. time",round(time()-t1, 3),'sec'

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import KFold
kf = KFold(n_splits=3)
print(kf)
count=0
for trainIndex, testIndex in kf.split(labels):
    features_train = [features[index] for index in trainIndex]
    features_test =  [features[index] for index in testIndex]
    labels_train =   [labels[index] for index in trainIndex]
    labels_test =    [labels[index] for index in testIndex]

clf = DecisionTreeClassifier(min_samples_split=6)
clf = clf.fit(features_train,labels_train)
pred = clf.predict(features_test)
accuracy = accuracy_score(labels_test,pred)
t1 = time()
print 'Accuracy',round(accuracy,2)
print "Precision: ",round(precision_score(labels_test,pred),2)
print "Recall: ", round(recall_score(labels_test,pred),2)
print "DecisionTree Clf algo. time",round(time()-t1, 3),'sec'

# # Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

ts.dump_classifier_and_data(clf, my_dataset, features_list)
ts.main()