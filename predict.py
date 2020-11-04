import sys
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import load_kickstarter_data, extract_json_data, clean_data, get_duration, get_blurb_length, currency_conversion, get_target, drop_columns, get_target_and_features, make_dummies, fix_skew, scale_features
RSEED = 50

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv)) 

#in an ideal world this would validated
# in terminal: python predict.py pickle_file.sav x_test.csv y_test.csv
model = sys.argv[1]
X_test_path = sys.argv[2]
y_test_path = sys.argv[3]

# load the model from disk
loaded_model = pickle.load(open(model, 'rb'))
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)

#feature eng on test data
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
    
# This is FUTURE WORK
# split categorical columns into dummies
#cat_columns=['country', 'category_main','category_sub']
#X_test = make_dummies(X_test, cat_columns)

# address skew   by applying logarithm  
num_columns = ['project_duration_days', 'blurb_length', 'usd_goal']
X_test = fix_skew(X_test, skewed=num_columns)

# scale numerical features
X_test = scale_features(X_test, num_columns)
print("Feature engineering on test data complete")

y_test_pred = loaded_model.predict(X_test)

print("Confusion Matrix: \n", 
confusion_matrix(y_test, y_test_pred)) 
    
print ("Accuracy : \n", 
accuracy_score(y_test, y_test_pred)*100) 
    
print("Report : \n", 
classification_report(y_test, y_test_pred)) 
