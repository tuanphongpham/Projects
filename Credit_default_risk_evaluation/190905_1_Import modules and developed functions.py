# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 01:49:30 2019

@author: Phong
"""

# Pandas and numpy for data manipulation
import pandas as pd
import numpy as np

import functools
from scipy.stats import kurtosis, skew

kurtosis_pearson = functools.partial(kurtosis, fisher=False)
skew_p = functools.partial(skew)
#std_p = functools.partial(std)
# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings from pandas
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')


import matplotlib.pyplot as plt
import lightgbm as lgb

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

# Memory management
import gc
import os
# os.chdir('C:\\Users\\Phong\\')

pd.set_option('display.max_columns', 500)
############## 1_ SPLIT TRAIN TEST DATA #############################################
def split_train_test(data, test_ratio):
    # data = housing
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


#train_set = pd.read_csv('application_train.csv')

#train_official, test_offcicial = split_train_test(train_set, 0.5)
#train, test = split_train_test(train_official, 0.2)
#gc.enable()
#del test_offcicial, train_set, train_official, 
#gc.collect() 

####. 1_Function to Aggregate Numeric Data
def agg_numeric(df, parent_var, df_name):
    """
    Groups and aggregates the numeric values in a child dataframe
    by the parent variable.
    
    Parameters
    --------
        df (dataframe): 
            the child dataframe to calculate the statistics on
        parent_var (string): 
            the parent variable used for grouping and aggregating
        df_name (string): 
            the variable used to rename the columns
        
    Return
    --------
        agg (dataframe): 
            a dataframe with the statistics aggregated by the `parent_var` for 
            all numeric columns. Each observation of the parent variable will have 
            one row in the dataframe with the parent variable as the index. 
            The columns are also renamed using the `df_name`. Columns with all duplicate
            values are removed. 
    
    """
    
    # Remove id variables other than grouping variable
    for col in df:
        if col != parent_var and 'SK_ID' in col:
            df = df.drop(columns = col)
            
    # Only want the numeric variables
    parent_ids = df[parent_var].copy()
    numeric_df = df.select_dtypes('number').copy()
    numeric_df[parent_var] = parent_ids

    # Group by the specified variable and calculate the statistics
    agg = numeric_df.groupby(parent_var).agg(['count', 'mean', 'max', 'min', 'sum','var','std', 'skew'])#]) # With 'std': problem with NA

    # Need to create new column names
    columns = []

    # Iterate through the variables names
    for var in agg.columns.levels[0]:
        if var != parent_var:
            # Iterate through the stat names
            for stat in agg.columns.levels[1]:
                # Make a new column name for the variable and stat
                columns.append('%s_%s_%s' % (df_name, var, stat))
    
    agg.columns = columns
    
    # Remove the columns with all redundant values
    _, idx = np.unique(agg, axis = 1, return_index=True)
    agg = agg.iloc[:, idx]
    
    return agg

#train = pd.read_csv('application_train.csv')#.head()
#test add = agg_numeric(bureau1, 'SK_ID_CURR', 'app')
#a=[2,1,0,2,0]
#skew(a)
#### 2. Function to calculate categorical counts
# normed count, which is the count for a category divided by the total counts for all categories in a categorical variable.
# counts: the occurrences  of each category in a categorical variable 

def agg_categorical(df, parent_var, df_name):
    """
    Aggregates the categorical features in a child dataframe
    for each observation of the parent variable.
    
    Parameters
    --------
    df : dataframe 
        The dataframe to calculate the value counts for.
        
    parent_var : string
        The variable by which to group and aggregate the dataframe. For each unique
        value of this variable, the final dataframe will have one row
        
    df_name : string
        Variable added to the front of column names to keep track of columns

    
    Return
    --------
    categorical : dataframe
        A dataframe with aggregated statistics for each observation of the parent_var
        The columns are also renamed and columns with duplicate values are removed.
        
    """
    
    # Select the categorical columns
    categorical = pd.get_dummies(df.select_dtypes('category'))

    # Make sure to put the identifying id on the column
    categorical[parent_var] = df[parent_var]

    # Groupby the group var and calculate the sum and mean
    categorical = categorical.groupby(parent_var).agg(['sum', 'count','mean'])
    
    column_names = []
    
    # Iterate through the columns in level 0
    for var in categorical.columns.levels[0]:
        # Iterate through the stats in level 1
        for stat in ['sum', 'count', 'mean']:
            # Make a new column name
            column_names.append('%s_%s_%s' % (df_name, var, stat))
    
    
    categorical.columns = column_names
    
    # Remove duplicate columns by values
    _, idx = np.unique(categorical, axis = 1, return_index = True)
    categorical = categorical.iloc[:, idx]
    
    return categorical

    # add1 = agg_categorical(bureau1 , parent_var = 'SK_ID_CURR', df_name = 'app')
# Function to Convert Data Types
# This will help reduce memory usage by using more efficient types for the variables: or example category is often a better type than object
import sys

def return_size(df):
    """Return size of dataframe in gigabytes"""
    return round(sys.getsizeof(df) / 1e9, 2)

def convert_types(df, print_info = False):
    
    original_memory = df.memory_usage().sum()
    
    # Iterate through each column
    for c in df:
        
        # Convert ids and booleans to integers
        if ('SK_ID' in c):
            df[c] = df[c].fillna(0).astype(np.int32)
            
        # Convert objects to category
        elif (df[c].dtype == 'object') and (df[c].nunique() < df.shape[0]):
            df[c] = df[c].astype('category')
        
        # Booleans mapped to integers
        elif list(df[c].unique()) == [1, 0]:
            df[c] = df[c].astype(bool)
        
        # Float64 to float32
        elif df[c].dtype == float:
            df[c] = df[c].astype(np.float32)
            
        # Int64 to int32
        elif df[c].dtype == int:
            df[c] = df[c].astype(np.int32)
        
    new_memory = df.memory_usage().sum()
    
    if print_info:
        print(f'Original Memory Usage: {round(original_memory / 1e9, 2)} gb.')
        print(f'New Memory Usage: {round(new_memory / 1e9, 2)} gb.')
        
    return df
######## Function to Calculate Missing Values

def missing_values_table(df, print_info = False):
    # Total missing values
    mis_val = df.isnull().sum()
    mis_val_percent = 100*df.isnull().sum()/len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis =1)
    
    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(columns={0:'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns  = mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:,1]!=0].sort_values('% of Total Values', ascending=False).round(1)
    if print_info:
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
                "There are " + str(mis_val_table_ren_columns.shape[0]) +
                  " columns that have missing values.")
        
        # Return the dataframe with missing information
    return mis_val_table_ren_columns

######### remove_missing_columns

def remove_missing_columns(train, test, threshold = 99):
    # Calculate missing stats for train and test (remember to calculate a percent!)
    train_miss = pd.DataFrame(train.isnull().sum())
    train_miss['percent'] = 100 * train_miss[0] / len(train)
    
    test_miss = pd.DataFrame(test.isnull().sum())
    test_miss['percent'] = 100 * test_miss[0] / len(test)
    
    # list of missing columns for train and test
    missing_train_columns = list(train_miss.index[train_miss['percent'] > threshold])
    missing_test_columns = list(test_miss.index[test_miss['percent'] > threshold])
    
    # Combine the two lists together
    missing_columns = list(set(missing_train_columns + missing_test_columns))
    
    # Print information
    print('There are %d columns with greater than %d%% missing values.' % (len(missing_columns), threshold))
    
    # Drop the missing columns and return
    train = train.drop(columns = missing_columns)
    test = test.drop(columns = missing_columns)
    
    return train, test

# Function to Aggregate Stats at the Client Level

def aggregate_client(df, group_vars, df_names):
    """Aggregate a dataframe with data at the loan level 
    at the client level
    
    Args:
        df (dataframe): data at the loan level
        group_vars (list of two strings): grouping variables for the loan 
        and then the client (example ['SK_ID_PREV', 'SK_ID_CURR'])
        names (list of two strings): names to call the resulting columns
        (example ['cash', 'client'])
        
    Returns:
        df_client (dataframe): aggregated numeric stats at the client level. 
        Each client will have a single row with all the numeric data aggregated
    """
    # Aggregate the numeric columns
    df_agg = agg_numeric(df, parent_var = group_vars[0], df_name=df_names[0])
    
    # Handle categorical variables
    if any(df.dtypes == 'category'):
        df_counts = agg_categorical(df, parent_var = group_vars[0], df_name = df_names[0])
        
        # Merge 2 dfs:
        df_by_loan1 = df_counts.merge(df_agg, on = group_vars[0], how = 'outer')
        gc.enable()
        del df_agg, df_counts
        gc.collect()
        
        # # Merge to get the client id in dataframe
        
        df_by_loan1 = df_by_loan1.merge(df[[group_vars[0], group_vars[1]]], on = group_vars[0], how = 'left')
        
        # remove the loan id
        
        df_by_loan1 = df_by_loan1.drop(columns= [group_vars[0]])
        # Aggregate numeric stats by column
        df_by_client = agg_numeric(df_by_loan1, parent_var = group_vars[1], df_name = df_names[1])
    # No categorical variables
    else:        
        df_by_loan1 = df_agg.merge(df[[group_vars[0], group_vars[1]]], on = group_vars[0], how ='left')
        gc.enable()
        del df_agg
        gc.collect()
        
        # Remove the loan id
        df_by_loan1 = df_by_loan1.drop(columns = [group_vars[0]])
        # Aggregate numeric stats by column
        df_by_client = agg_numeric(df_by_loan1, parent_var = group_vars[1], df_name = df_names[1])
        
    # Memory management
    gc.enable()
    del df, df_by_loan1
    gc.collect()
    
    return df_by_client

### Function to find common elements between 2 lists:
def common(a,b):
    '''
    Function to find common between 2 lists
    Inputs: 2 list a, b
    Out put: list common
    '''
    com = []
    #for i in range(len(a)):
    #    if a[i] in b:
    #        com.append(str(a[i]))
    com =list(set(a)&set(b))
    return com