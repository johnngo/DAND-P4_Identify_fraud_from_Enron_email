#!/usr/bin/python
'''
Please refer to the accompany notebook.

   enronProject.ipynb
'''
import sys
import pickle
import numpy as np 
import pandas as pd 
from time import time
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")


from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    enronData = pickle.load(data_file)

# Convert data dictionary into a pandas dataframe, for ease of manipulation

df = pd.DataFrame.from_dict(enronData, orient = 'index')

df.replace('NaN', 0, inplace = True)

# features_list = ['poi','salary'] # You will need to use more features


### Task 2: Remove outliers

df = df.drop(['TOTAL', 'THE TRAVEL AGENCY IN THE PARK'])


### Task 3: Create new feature(s)

# df.replace(0, np.nan, inplace = True)

# df['f_to_poi'] = df['from_this_person_to_poi']/df['from_messages']

# df['f_from_poi'] = df['from_poi_to_this_person']/df['to_messages']


# Replace np.nan with NaN (for compatibility with feature_format.py)

# df.replace(np.nan, 'NaN', inplace = True)

# create a dictionary from the dataframe
# df_dict = df.to_dict('index')

### Store to my_dataset for easy export below.
# my_dataset = df_dict

#3 create new feature
#new features: f_to_poi = number of emails sent to POIs, f_from_poi = number of emails received  from POI

def dict_to_list(key,normalizer):
    new_list=[]

    for i in enronData:
        if enronData[i][key]=="NaN" or enronData[i][normalizer]=="NaN":
            new_list.append(0.)
        elif enronData[i][key]>=0:
            new_list.append(float(enronData[i][key])/float(enronData[i][normalizer]))
    return new_list

### create two lists of new features
f_from_poi = dict_to_list("from_poi_to_this_person","to_messages")
f_to_poi = dict_to_list("from_this_person_to_poi","from_messages")

### insert new features into enronData
count=0
for i in enronData:
    enronData[i]["f_from_poi"]=f_from_poi[count]
    enronData[i]["f_to_poi"]=f_to_poi[count]
    count +=1

    
features_list = ["poi", "f_from_poi", "f_to_poi"]    
    ### store to my_dataset for easy export below
my_dataset = enronData



# features_list is a list of strings, each of which is a feature name.
# The first feature must be "poi".

# features_list = ['poi','salary', 'deferral_payments', 'total_payments', 
# 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 
# 'total_stock_value', 'expenses', 'exercised_stock_options', 'other',
# 'long_term_incentive', 'restricted_stock', 'director_fees','to_messages',
# 'from_poi_to_this_person', 'from_messages','from_this_person_to_poi', 
# 'shared_receipt_with_poi','f_from_poi','f_to_poi']

### features_list is a list of strings, each of which is a feature name
### first feature must be "poi", as this will be singled out as the label
features_list = ["poi", "exercised_stock_options", "f_from_poi", "f_to_poi", "shared_receipt_with_poi"]


### store to my_dataset for easy export below
my_dataset = enronData


### these two lines extract the features specified in features_list
### and extract them from enronData, returning a numpy array
data = featureFormat(my_dataset, features_list)


### split into labels and features (this line assumes that the first
### feature in the array is the label, which is why "poi" must always
### be first in features_list
labels, features = targetFeatureSplit(data)


### machine learning goes here!
### please name your classifier clf for easy export below

### deploying feature selection
from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

        
        
### use KFold for split and validate algorithm
from sklearn.cross_validation import KFold
kf=KFold(len(labels),3)
for train_indices, test_indices in kf:
    #make training and testing sets
    features_train= [features[ii] for ii in train_indices]
    features_test= [features[ii] for ii in test_indices]
    labels_train=[labels[ii] for ii in train_indices]
    labels_test=[labels[ii] for ii in test_indices]        



from sklearn.tree import DecisionTreeClassifier

t0 = time()

clf = DecisionTreeClassifier()
clf.fit(features_train,labels_train)
score = clf.score(features_test,labels_test)
print 'accuracy before tuning ', score

print "Decision tree algorithm time:", round(time()-t0, 3), "s"


### use manual tuning parameter min_samples_split
t0 = time()
clf = DecisionTreeClassifier(min_samples_split = 3)
clf = clf.fit(features_train,labels_train)
pred= clf.predict(features_test)
print("done in %0.3fs" % (time() - t0))

acc=accuracy_score(labels_test, pred)

print "Validating algorithm:"
print "accuracy after tuning = ", acc

# function for calculation ratio of true positives
# out of all positives (true + false)
print 'precision = ', precision_score(labels_test,pred)

# function for calculation ratio of true positives
# out of true positives and false negatives
print 'recall = ', recall_score(labels_test,pred)

### Extract features and labels from dataset for local testing
# data = featureFormat(my_dataset, features_list, remove_NaN=True, remove_all_zeroes=True, 
#                      remove_any_zeroes=True, sort_keys = True)

# data = featureFormat(my_dataset, features_list, sort_keys = True)

# labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
# from sklearn.tree import DecisionTreeClassifier

# clf = DecisionTreeClassifier(random_state=42)


# clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=4,
#             max_features=7, max_leaf_nodes=None, min_impurity_decrease=0.0,
#             min_impurity_split=None, min_samples_leaf=1,
#             min_samples_split=2, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='best')

# clf = DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=3,
#             max_features=7, max_leaf_nodes=8, min_impurity_decrease=0.0,
#             min_impurity_split=None, min_samples_leaf=1,
#             min_samples_split=4, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='random')


# clf = DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=4,
#             max_features=2, max_leaf_nodes=None, min_impurity_decrease=0.0,
#             min_impurity_split=None, min_samples_leaf=1,
#             min_samples_split=6, min_weight_fraction_leaf=0.0,
#             presort=False, random_state=None, splitter='best')

# clf.fit(features,labels)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

