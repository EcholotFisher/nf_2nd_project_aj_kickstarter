# import packages 
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample

import glob
import json
RSEED = 50
#from time import time
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import fbeta_score

def load_kickstarter_data(datapath):
    '''datapath = location of csv files to be loaded'''
    # List with the names of all the csv files in the path
    csv_files = glob.glob(datapath+'/*.csv')

    print(f'Total files: {len(csv_files)}')

    # Loop through the files
    for file_idx, csv_file in enumerate(csv_files): 
        # create dataframe from 1st csv       
        if file_idx == 0:
            df_ks = pd.read_csv(csv_file)
            print(f'File number {file_idx + 1} added to dataframe')
        else:
            # create dataframe from idx csv
            df = pd.read_csv(csv_file)
            # check files are all in same
            if  np.all(df.columns == df_ks.columns) == False:
                print(f'Column format of {csv_file} does not match {csv_files[0]}. Please check and try again')
                return
            else:
                # append to initial dataframe                   
                df_ks = pd.concat([df_ks, df], axis=0, ignore_index=True)       
                print(f'File number {file_idx + 1} added to dataframe')
                
    print('File import done')
    return df_ks

def extract_json_data(data):
    ''' This function extracts specific sub fields from json files embedded in columns of a dataframe
        data: dataframe containing column with json data'''
    data['category_name'] = pd.DataFrame.from_dict([json.loads(data["category"][i])['name'] for i in range(data.shape[0])])
    data['category_slug'] = pd.DataFrame([json.loads(data["category"][i])['slug'] for i in range(data.shape[0])])
    # Split slug into main category and sub category
    data[['category_main','category_sub']] = data.category_slug.str.split(pat='/', n=1, expand=True)
    data.drop(labels = ['category','category_slug'], axis=1, inplace=True)
    
    print('json columns extracted')
    return data

def get_duration(data):
    #Convert from unix time stamp to more readable time format
    data['converted_deadline'] = pd.to_datetime(data['deadline'], unit='s')
    data['converted_launched_at'] = pd.to_datetime(data['launched_at'], unit='s')
    #Create project duration variable
    data['project_duration_days'] = (data['converted_deadline'] - data['converted_launched_at']).dt.days
    # Drop redundant columns
    data.drop(columns=['deadline', 'launched_at'], inplace=True)
    return data

def get_target(data,target='state', new_target_var='success', success_label='successful'):
    '''
    creates a dummy variable out of the state to be used as dependant variable
    '''
    #data('success') = data['state'].apply(lambda x: 1 if x == 'successful' else 0)
    data[new_target_var] = data[target].apply(lambda x: 1 if x == success_label else 0)
    return data

def clean_data(data):
    ''' Clean out the outliers and open campaigns data'''
    # print(df.goal.quantile(.98))
    goal_98_quantile = 200000
    data = data[data.state != 'live'] # change this to sucessful and others if you want cancelled etc. 
    data = data[data.goal < goal_98_quantile]
    return data

def get_target_and_features(data, target_var='success'):
    '''
    Function that splits dataset into target and feature dataframes
    '''
    #target = data['success']
    target = data[target_var]
    data.drop([target_var,'state'], axis = 1, inplace=True)
    features = data

    print('target and features split is done')
    return target, features

def currency_conversion(data):
    ''' Convert the currency of all projects to USD. '''
    # We use static_usd_rate since this is what was used for usd_pledged:
        # df['implied_fx_rate'] = df['usd_pledged'] / df['pledged'] ==  df['static_usd_rate']
    data['usd_goal'] = data['goal'] * data['static_usd_rate']
    # drop goal and static_usd_rate to remove redundant data
    data.drop(columns=['goal','static_usd_rate'], inplace=True)
    return data

def make_dummies(data, columns=['country', 'category_main','category_sub']):
    '''It does what it says :)'''
    data = pd.get_dummies(data, columns=columns, drop_first=True) #Avoid dummy trap   
    return data

def get_blurb_length(data):
    '''
    Create new features: Blurb length
    '''
    data['blurb_length'] = data.blurb.apply(lambda x: len(str(x)))
    return data

def fix_skew(data, skewed=['usd_goal']):
    '''Log-transform the skewed features'''
    data[skewed] = data[skewed].apply(lambda x: np.log(x + 1))
    return data

def scale_features(data, num_columns=['usd_goal','project_duration_days','blurb_length']):
    ''' Initialize a scaler, then apply it to the features'''    
    scaler = MinMaxScaler()
    data[num_columns] = scaler.fit_transform(data[num_columns])    
    return data

def drop_columns(data):
    '''remove unnecessary columns'''
    # Drop due to many missing values
    data.drop(columns = ['friends', 'is_backing', 'is_starred', 'permissions'], inplace=True)
    # Some Json strings varariables with unusable or already used data
    data.drop(columns = ['creator', 'location', 'photo', 'profile', 'slug', 'urls'], inplace=True)
    # Columns that are not specific to the campaign or are redundant or are technical data unrelated to campaign
    data.drop(columns = ['created_at','currency', 'currency_symbol', 'currency_trailing_code', 
                     'current_currency', 'disable_communication',
                     'is_starrable', 'source_url', 'spotlight', 'staff_pick', 
                     'usd_type', 'state_changed_at','fx_rate'], inplace=True)
    # drop columns due to being linked to dependent variable which would not be known in advance
    data.drop(columns = ['backers_count', 'converted_pledged_amount', 'pledged', 'usd_pledged','id'], inplace=True) # to be checked 'backers_count'
    # drop columns that are not used                
    data.drop(columns = ['blurb', 'name', 'converted_deadline', 'converted_launched_at','category_name',], inplace=True) 
    return data



def rebalance(X_train, y_train):
    
    # concatenate our training data back together
    X = pd.concat([X_train, y_train], axis=1)
    print(f'initial split \n{X.success.value_counts()}')
    # separate minority and majority classes
    fail = X[X.success==0]
    succeed = X[X.success==1]
  
    # upsample minority
    fail_upsampled = resample(fail,
                              replace=True, # sample with replacement
                              n_samples=len(succeed), # match number in majority class
                              random_state=RSEED) # reproducible results

    # combine majority and upsampled minority
    upsampled = pd.concat([succeed, fail_upsampled])

    # check new class counts
    print(f'final split \n{upsampled.success.value_counts()}')
    y_train = upsampled.success
    X_train = upsampled.drop('success', axis=1)
    return X_train, y_train
