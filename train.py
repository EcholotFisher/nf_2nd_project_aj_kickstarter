import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import load_kickstarter_data, extract_json_data, clean_data, get_duration, get_blurb_length, currency_conversion
from feature_engineering import get_target, drop_columns, get_target_and_features, make_dummies, fix_skew, scale_features, rebalance
RSEED = 50

# Read data
df = load_kickstarter_data('kickstarter/data')

# Extract category data from json
df = extract_json_data(df)

# data cleaning: drop campaigns that are still ongoing and outliers 
df = clean_data(df)

# FUTURE WORK: move this to after the test train split and 
#   deal with mismatched category_sub unique values in train and test data
# split categorical columns into dummies


# encode target variable 'state' to numerical values, success is 1 all others are fail and 0
df = get_target(df,target='state', new_target_var='success', success_label='successful')

# Split the data into features and target label
target, features = get_target_and_features(df)

# split data for testing
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.2, random_state = RSEED)

## in order to exemplify how the predict will work.. we will save the y_train
print("Saving test data in the data folder")
X_test.to_csv("test_data/X_test.csv", index=False)
y_test.to_csv("test_data/y_test.csv", index=False)

print("Feature engineering on train data starting...")
# convert unix timestamps and calculate campaign duration
X_train = get_duration(X_train)

# Get blurb length
X_train = get_blurb_length(X_train)

#create goal data in single currency
X_train = currency_conversion(X_train)

# drop unnecessary columns
X_train = drop_columns(X_train)

print('get_dummies')
cat_columns=['country', 'category_main','category_sub']
X_train = make_dummies(X_train, cat_columns)    

# address skew   by applying logarithm  
num_columns = ['project_duration_days', 'blurb_length', 'usd_goal']
X_train = fix_skew(X_train, skewed=num_columns)

# scale numerical features
X_train = scale_features(X_train, num_columns)

# resample
X_train, y_train = rebalance(X_train, y_train)
print("Feature engineering on train data complete")

# model
print("Training a decision tree classifier")
clf_tree = DecisionTreeClassifier(criterion = "gini", 
            max_depth=3, min_samples_leaf=5) 
    # Performing training 
clf_tree.fit(X_train, y_train) 
# Create predictions using simple model - decision tree
y_train_pred = clf_tree.predict(X_train) 
# show results
print("Confusion Matrix: \n", 
confusion_matrix(y_train, y_train_pred)) 
    
print ("Accuracy : \n", 
accuracy_score(y_train, y_train_pred)*100) 
    
print("Report : \n", 
classification_report(y_train, y_train_pred)) 

#saving the model
print("Saving Decision tree model in the model folder")
filename = 'models/DecisionTreeClassifier_model.sav'
pickle.dump(clf_tree, open(filename, 'wb'))


# Create predictions using simple model - decision tree
print("Feature engineering on test data starting...")
# convert unix timestamps and calculate campaign duration
X_test = get_duration(X_test)

# Get blurb length
X_test = get_blurb_length(X_test)

#create goal data in single currency
X_test = currency_conversion(X_test)

# drop unnecessary columns
X_test = drop_columns(X_test)
    
# split categorical columns into dummies
cat_columns=['country', 'category_main','category_sub']
X_test = make_dummies(X_test, cat_columns)

# address skew   by applying logarithm  
num_columns = ['project_duration_days', 'blurb_length', 'usd_goal']
X_test = fix_skew(X_test, skewed=num_columns)

# scale numerical features
X_test = scale_features(X_test, num_columns)

print("Feature engineering on test data complete")

y_pred = clf_tree.predict(X_test) 
# show results

print("Confusion Matrix: \n", 
confusion_matrix(y_test, y_pred)) 
    
print ("Accuracy : \n", 
accuracy_score(y_test, y_pred)*100) 
    
print("Report : \n", 
classification_report(y_test, y_pred)) 


