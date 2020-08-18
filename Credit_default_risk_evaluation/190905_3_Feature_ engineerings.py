# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 01:35:32 2019

@author: Phong
"""
def split_train_test(data, test_ratio):
    # data = housing
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
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



# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#

#############################################################################################################################################################################
######################## START FEATURES ENGINEERING HERE #################################################################################################################
#############################################################################################################################################################################
# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX#
    





##################################### 0. Application train Data set #################################################################################################
    

app_train = pd.read_csv('application_train.csv')
############### Create Domain Knowledge features
# CREDIT_INCOME_PERCENT: the percentage of the credit amount relative to a client's income
# ANNUITY_INCOME_PERCENT: the percentage of the loan annuity relative to a client's income
# CREDIT_TERM: the length of the payment in months (since the annuity is the monthly amount due
# DAYS_EMPLOYED_PERCENT: the percentage of the days employed relative to the client's age

app_train['CREDIT_INCOME_PERCENT'] = app_train['AMT_CREDIT']/app_train['AMT_INCOME_TOTAL']
app_train['ANNUITY_INCOME_PERCENT'] = app_train['AMT_ANNUITY']/app_train['AMT_INCOME_TOTAL']
app_train['CREDIT_TERM'] = app_train['AMT_ANNUITY']/app_train['AMT_CREDIT']
app_train['DAYS_EMPLOYED_PERCENT'] = app_train['DAYS_EMPLOYED']/app_train['DAYS_BIRTH']
app_train['INCOME_PER_PERSON'] = app_train['AMT_INCOME_TOTAL'] / app_train['CNT_FAM_MEMBERS']
app_train['PAYMENT_RATE'] = app_train['AMT_ANNUITY'] / app_train['AMT_CREDIT']

train_set, test_set = split_train_test(train_set, 0.2)

train_set.to_csv('train_set.csv', index = False)
test_set.to_csv('test_set.csv', index = False)



#############################################################################################################################################################################
######################## STAGE 1: Features from application_train, bureau and bureau_balance ################################################################################
########### 1_ Bureau Data Set ###############################################################################################################################################
train = pd.read_csv('train_set.csv')
test = pd.read_csv('test_set.csv')
bureau = pd.read_csv('bureau.csv')#.head(50000)
bureau = convert_types(bureau, print_info = True)
bureau.info()

previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index = False)['SK_ID_BUREAU'].count().rename(columns ={'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head()
train = train.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')
train['previous_loan_counts'] = train['previous_loan_counts'].fillna(0)
train.info()

test = test.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')
test['previous_loan_counts'] = test['previous_loan_counts'].fillna(0)
test.info()

bureau['CREDIT_DAY_OVERDUE_TIME_DAYS_CREDIT'] = bureau['CREDIT_DAY_OVERDUE'] * bureau['DAYS_CREDIT']
bureau_by_client = aggregate_client(bureau, group_vars =['SK_ID_BUREAU', 'SK_ID_CURR'], df_names = ['bureau', 'client'])

list(bureau_by_client.columns)

train=train.merge(bureau_by_client, on = 'SK_ID_CURR', how = 'left' )

test = test.merge(bureau_by_client, on = 'SK_ID_CURR', how = 'left' )
 
gc.enable()
del bureau , bureau_by_client 
gc.collect()
train, test = remove_missing_columns(train, test)
train.info()
########### 2_ Bureau_balance Data Set ######################################################################################################################################
bureau_balance = pd.read_csv('bureau_balance.csv')
bureau_balance.head()
bureau = pd.read_csv('bureau.csv')[['SK_ID_BUREAU', 'SK_ID_CURR']]
bureau_balance = bureau_balance.merge(bureau, on ='SK_ID_BUREAU', how = 'left')

bureau_balance = convert_types(bureau_balance, print_info = True)
bureau_balance.info()

bureau_balance_by_client = aggregate_client(bureau_balance, group_vars =['SK_ID_BUREAU', 'SK_ID_CURR'], df_names = ['bureau_balance', 'client'])

bureau_balance_by_client.head()

train=train.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')

test = test.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del bureau_balance_by_client, bureau_balance, bureau
gc.collect()
train, test = remove_missing_columns(train, test)
train.info()

train.to_csv('train_after_stage1.csv', index = False)
test.to_csv('test_after_stage1.csv', index = False)

train.info()
'TARGET' in list(train.columns)
'TARGET' in list(test.columns)
set(list(train.columns)) - set(list(test.columns))
test.info()

##################################### STAGE 2 ###################################################################################################################
################ All data except for ata from Installment Payments ##############################################################################################
########### 3_ previous_application Data Set #############################################################################################################################

previous=pd.read_csv('previous_application.csv')
previous = convert_types(previous, print_info=True)
previous.head()

previous_agg = agg_numeric(previous, 'SK_ID_CURR', 'previous')
previous_agg.shape # 37 columns -> 70 columns


previous_counts = agg_categorical(previous, 'SK_ID_CURR', 'previous')
previous_counts.shape # 37 columns -> 285 columns
list(previous_counts.columns)

# train = pd.read_csv('train_after_stage1.csv')
train = convert_types(train)

# test =pd.read_csv('test_after_stage1.csv')

test.info()
test = convert_types(test)

# Merge new features into train and test
train = train.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
train = train.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')

test = test.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
test = test.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')

# Remove variables to free memory
gc.enable()
del previous, previous_agg, previous_counts
gc.collect()

train, test = remove_missing_columns(train, test)

########### 4_ Monthly Cash Data Set####################################################################################################################################
cash = pd.read_csv('POS_CASH_balance.csv')
cash = convert_types(cash, print_info = True)
cash.head()
cash.info()

cash_by_client = aggregate_client(cash, group_vars =['SK_ID_PREV', 'SK_ID_CURR'], df_names =['cash', 'client'])
cash_by_client.info()
cash_by_client.head()

print('Cash by client Shape: ', cash_by_client.shape)
train = train.merge(cash_by_client, on ='SK_ID_CURR', how ='left')
test = test.merge(cash_by_client, on = 'SK_ID_CURR', how ='left')

gc.enable()
del cash, cash_by_client
gc.collect()

train, test = remove_missing_columns(train, test)

########### 5_ Monthly Credit Data Set ##################################################################################################################################

credit = pd.read_csv('credit_card_balance.csv')
credit = convert_types(credit, print_info = True)
credit.info()
credit.head()

credit_by_client = aggregate_client(credit, group_vars=['SK_ID_PREV', 'SK_ID_CURR'], df_names=['credit','client'])
credit_by_client.head()

train = train.merge(credit_by_client, on = 'SK_ID_CURR', how ='left')
test = test.merge(credit_by_client, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del credit, credit_by_client
gc.collect()
train, test = remove_missing_columns(train, test)
##################################################################################################################################################################
########################################## STAGE 3: include data from Installment Payments ######################################################################
########### 6_Installment Payments Data Set #############################################################################################################################

installments =pd.read_csv('installments_payments.csv')
installments = convert_types(installments, print_info = True)
installments.info()
installments.head()


installments_by_client = aggregate_client(installments, group_vars =['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['installments', 'client'])

installments_by_client.head()

train=train.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left' )

test = test.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left' )

gc.enable()
del installments, installments_by_client
gc.collect()
train, test = remove_missing_columns(train, test)
train.info()
test.info()

test['TARGET'] = test_labels

print(f'Final training size: {return_size(train)}')
print(f'Final testing size: {return_size(test)}')

train.to_csv('train_after_stage3.csv', index = False)
test.to_csv('test_after_stage3.csv', index = False)
set(list(train.columns)) - set(list(test.columns))


################################ STAGE 4 ########################################################################################################################
############################## Create more polinomial features for data after stage 3 ##################################################################################################
################################# ###############################################################################################################################
train = pd.read_csv('train_after_stage3.csv')
app_train = train # convert_types(train)

test =pd.read_csv('test_after_stage3.csv')

app_test = test#convert_types(test) 


######################## Create the Polynomial Features from most important 8 feactors:



poly_features = app_train[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH',
                           'AMT_CREDIT','AMT_ANNUITY', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_EMPLOYED']]
poly_features_test = app_test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH',
                           'AMT_CREDIT','AMT_ANNUITY', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_EMPLOYED']]


# Imputer for handling the missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'median')
poly_target = app_train['TARGET']

poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)

# Create the polynomial object with specified degree: add 212 features
from sklearn.preprocessing import PolynomialFeatures
poly_transformer = PolynomialFeatures(degree = 3)
poly_transformer.fit(poly_features)
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features.shape, poly_features_test.shape)
column = poly_transformer.get_feature_names(input_features =['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'DAYS_BIRTH',
                           'AMT_CREDIT','AMT_ANNUITY', 'AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_EMPLOYED'])

# Create a dataframe of the features
poly_features = pd.DataFrame(poly_features, columns = column)
poly_features['SK_ID_CURR'] = app_train['SK_ID_CURR']
poly_features_test = pd.DataFrame(poly_features_test, columns = column)
poly_features_test['SK_ID_CURR'] = app_test['SK_ID_CURR']
poly_features_test.shape
poly_features.shape
# Merger to app_train and app_test:
app_train = app_train.merge(poly_features, on = 'SK_ID_CURR', how = 'left')
app_train.shape
app_test = app_test.merge(poly_features_test, on = 'SK_ID_CURR', how ='left')
app_test.shape



app_train.to_csv('train_after_stage4.csv', index = False)
app_test.to_csv('test_after_stage4.csv', index = False)

#####################################################################################################################################################################
############### ROUND 2: Create More Knowledge Features and Time Features ##########################################################################################
###################################################################################################################################################################################################

############# 0.Application_train Data Set #################################################################################################################################

import pandas as pd
#df = pd.read_csv('application_train.csv')
df.head()
df = train 
df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

# Time features;
df['train_NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
df['train_DAYS_EMPLOYED - DAYS_BIRTH'] = df['DAYS_EMPLOYED'] - df['DAYS_BIRTH']
df['train_NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
df['train_NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
df['train_NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
df['train_NEW_PHONE_TO_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
df['train_EXT_SOURCE_1 / DAYS_BIRTH'] = df['EXT_SOURCE_1'] / df['DAYS_BIRTH']
df['train_EXT_SOURCE_2 / DAYS_BIRTH'] = df['EXT_SOURCE_2'] / df['DAYS_BIRTH']
df['train_EXT_SOURCE_3 / DAYS_BIRTH'] = df['EXT_SOURCE_3'] / df['DAYS_BIRTH']
# Knowledge features:
df['train_NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
df['train_NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
df['train_NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
df['train_NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
df['train_NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
df['train_NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
df['train_NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
df['train_NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
df['train_NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
df= df.drop(dropcolum,axis=1)

train = df

######## 1. bureau #################################################################################################################################################
# bureau: information about client's previous loans with other financial institutions reported to Home Credit. Each previous loan has its own row.
# bureau_balance: monthly information about the previous loans.
# Each month has its own row.
https://www.kaggle.com/shanth84/home-credit-bureau-data-feature-engineering

# UNDERSTANDING OF VARIABLES¶
# CREDIT_ACTIVE - Current status of a Loan - Closed/ Active (2 values)

# CREDIT_CURRENCY - Currency in which the transaction was executed - Currency1, Currency2, Currency3, Currency4 ( 4 values)

# CREDIT_DAY_OVERDUE - Number of overdue days

# CREDIT_TYPE - Consumer Credit, Credit card, Mortgage, Car loan, Microloan, Loan for working capital replemishment, Loan for Business development, Real estate loan, Unkown type of laon, Another type of loan. Cash loan, Loan for the purchase of equipment, Mobile operator loan, Interbank credit, Loan for purchase of shares ( 15 values )

# DAYS_CREDIT - Number of days ELAPSED since customer applied for CB credit with respect to current application Interpretation - Are these loans evenly spaced time intervals? Are they concentrated within a same time frame?

# DAYS_CREDIT_ENDDATE - Number of days the customer CREDIT is valid at the time of application CREDIT_DAY_OVERDUE - Number of days the customer CREDIT is past the end date at the time of application

# AMT_CREDIT_SUM - Total available credit for a customer AMT_CREDIT_SUM_DEBT - Total amount yet to be repayed
# AMT_CREDIT_SUM_LIMIT - Current Credit that has been utilized
# AMT_CREDIT_SUM_OVERDUE - Current credit payment that is overdue
# CNT_CREDIT_PROLONG - How many times was the Credit date prolonged
bureau = pd.read_csv('bureau.csv')
bureau = convert_types(bureau, print_info = True)
bureau.info()
# FEATURE 1 - NUMBER OF PAST LOANS PER CUSTOMER
previous_loan_counts = bureau.groupby('SK_ID_CURR', as_index = False)['SK_ID_BUREAU'].count().rename(columns ={'SK_ID_BUREAU': 'previous_loan_counts'})
previous_loan_counts.head()
train = train.merge(previous_loan_counts, on = 'SK_ID_CURR', how = 'left')
train['previous_loan_counts'] = train['previous_loan_counts'].fillna(0)
train.info()

# bureau_time_features =bureau[['SK_ID_CURR','DAYS_CREDIT_UPDATE', 'DAYS_ENDDATE_FACT', 'DAYS_CREDIT_ENDDATE',
#                       'CREDIT_DAY_OVERDUE', 'DAYS_CREDIT']]
# New time features
bureau['bureau_CREDIT_DAY_OVERDUE_TIME_DAYS_CREDIT'] = bureau['CREDIT_DAY_OVERDUE'] * bureau['DAYS_CREDIT']
bureau['bureau_AMT_CREDIT_SUM - AMT_CREDIT_SUM_DEBT'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_DEBT']
bureau['bureau_AMT_CREDIT_SUM - AMT_CREDIT_SUM_LIMIT'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_LIMIT']
bureau['bureau_AMT_CREDIT_SUM - AMT_CREDIT_SUM_OVERDUE'] = bureau['AMT_CREDIT_SUM'] - bureau['AMT_CREDIT_SUM_OVERDUE']

bureau['bureau_DAYS_CREDIT - CREDIT_DAY_OVERDUE'] = bureau['DAYS_CREDIT'] - bureau['CREDIT_DAY_OVERDUE']
bureau['bureau_DAYS_CREDIT - DAYS_CREDIT_ENDDATE'] = bureau['DAYS_CREDIT'] - bureau['DAYS_CREDIT_ENDDATE']
bureau['bureau_DAYS_CREDIT - DAYS_ENDDATE_FACT'] = bureau['DAYS_CREDIT'] - bureau['DAYS_ENDDATE_FACT']
bureau['bureau_DAYS_CREDIT_ENDDATE - DAYS_ENDDATE_FACT'] = bureau['DAYS_CREDIT_ENDDATE'] - bureau['DAYS_ENDDATE_FACT']
bureau['bureau_DAYS_CREDIT_UPDATE - DAYS_CREDIT_ENDDATE'] = bureau['DAYS_CREDIT_UPDATE'] - bureau['DAYS_CREDIT_ENDDATE']

# Conduct Aggregation
bureau_by_client = aggregate_client(bureau, group_vars =['SK_ID_BUREAU', 'SK_ID_CURR'], df_names = ['bureau', 'client'])

list(bureau_by_client.columns)

train=train.merge(bureau_by_client, on = 'SK_ID_CURR', how = 'left' )
test=test.merge(bureau_by_client, on = 'SK_ID_CURR', how = 'left' )

gc.enable()
del bureau , bureau_by_client 
gc.collect()
train = remove_missing_columns(train)
train.info()
train.to_csv('train_after_stage5_1.csv', index = False)
test.to_csv('test_after_stage5_1.csv', index = False)
#test_labels = test['TARGET']
#test_labels.to_csv('test_labels.csv', index = False)
#test1 = test.drop(columns =['TARGET'])
########### 2. Bureau_balance Data Set #########################################################################################################################################
# bureau: information about client's previous loans with other financial institutions reported to Home Credit. Each previous loan has its own row.
bureau_balance = pd.read_csv('bureau_balance.csv')
bureau_balance.head()
bureau = pd.read_csv('bureau.csv')[['SK_ID_BUREAU', 'SK_ID_CURR']]
bureau_balance = bureau_balance.merge(bureau, on ='SK_ID_BUREAU', how = 'left')

bureau_balance = convert_types(bureau_balance, print_info = True)
bureau_balance.info()

bureau_balance_by_client = aggregate_client(bureau_balance, group_vars =['SK_ID_BUREAU', 'SK_ID_CURR'], df_names = ['bureau_balance', 'client'])

bureau_balance_by_client.head()

train=train.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')
test=test.merge(bureau_balance_by_client, on = 'SK_ID_CURR', how = 'left')

gc.enable()
del bureau_balance_by_client, bureau_balance, bureau
gc.collect()
train, test = remove_missing_columns(train, test)
train.info()

train.to_csv('train_after_stage2_0.csv', index = False) # 970 features
test.to_csv('test_after_stage2_0.csv', index = False)


########### 3.previous_application Data Set #######################################################################################################################################################################################################################

previous=pd.read_csv('previous_application.csv')
previous = convert_types(previous, print_info=True)
previous.head()

previous['prev AMT_APPLICATION / AMT_CREDIT'] = previous['AMT_APPLICATION'] / previous['AMT_CREDIT']
previous['prev AMT_APPLICATION - AMT_CREDIT'] = previous['AMT_APPLICATION'] - previous['AMT_CREDIT']
previous['prev AMT_APPLICATION - AMT_GOODS_PRICE'] = previous['AMT_APPLICATION'] - previous['AMT_GOODS_PRICE']
previous['prev AMT_GOODS_PRICE - AMT_CREDIT'] = previous['AMT_GOODS_PRICE'] - previous['AMT_CREDIT']
previous['prev DAYS_FIRST_DRAWING - DAYS_FIRST_DUE'] = previous['DAYS_FIRST_DRAWING'] - previous['DAYS_FIRST_DUE']
previous['prev DAYS_TERMINATION less -500'] = (previous['DAYS_TERMINATION'] < -500).astype(int)

previous_agg = agg_numeric(previous, 'SK_ID_CURR', 'previous')
previous_agg.shape # 37 columns -> 70 columns


previous_counts = agg_categorical(previous, 'SK_ID_CURR', 'previous')
previous_counts.shape # 37 columns -> 285 columns
list(previous_counts.columns)

# train = pd.read_csv('train_after_stage1.csv')
train = convert_types(train)

# Merge new features into train and test
#train = train.merge(previous_counts, on ='SK_ID_CURR', how = 'left')
train = train.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')
test = test.merge(previous_agg, on = 'SK_ID_CURR', how = 'left')

# Remove variables to free memory
gc.enable()
del previous, previous_agg, previous_counts
gc.collect()

train, test = remove_missing_columns(train, test)


########### 4_ Monthly Cash Data Set #######################################################################################################################################################################################################################

cash = pd.read_csv('POS_CASH_balance.csv')
cash = convert_types(cash, print_info = True)
cash.head()
cash.info()

# Replace some outliers
cash.loc[cash['CNT_INSTALMENT_FUTURE'] > 60, 'CNT_INSTALMENT_FUTURE'] = np.nan
    
# Some new features
cash['pos CNT_INSTALMENT more CNT_INSTALMENT_FUTURE'] = (cash['CNT_INSTALMENT'] > cash['CNT_INSTALMENT_FUTURE']).astype(int)




cash_by_client = aggregate_client(cash, group_vars =['SK_ID_PREV', 'SK_ID_CURR'], df_names =['cash', 'client'])
cash_by_client.info()
cash_by_client.head()

print('Cash by client Shape: ', cash_by_client.shape)
train = train.merge(cash_by_client, on ='SK_ID_CURR', how ='left')

gc.enable()
del cash, cash_by_client
gc.collect()

train, test= remove_missing_columns(train, test)

########### 5_ Monthly Credit Data Set #######################################################################################################################################################################################################################


credit = pd.read_csv('credit_card_balance.csv')
credit = convert_types(credit, print_info = True)
credit.info()
credit.head()

 # Replace some outliers
credit.loc[credit['AMT_PAYMENT_CURRENT'] > 4000000, 'AMT_PAYMENT_CURRENT'] = np.nan
credit.loc[credit['AMT_CREDIT_LIMIT_ACTUAL'] > 1000000, 'AMT_CREDIT_LIMIT_ACTUAL'] = np.nan

# Some new features
credit['credit_card_ missing'] = credit.isnull().sum(axis = 1).values
credit['credit_card_SK_DPD - MONTHS_BALANCE'] = credit['SK_DPD'] - credit['MONTHS_BALANCE']
credit['credit_card_SK_DPD_DEF - MONTHS_BALANCE'] = credit['SK_DPD_DEF'] - credit['MONTHS_BALANCE']
credit['credit_card_SK_DPD - SK_DPD_DEF'] = credit['SK_DPD'] - credit['SK_DPD_DEF']
    
credit['credit_card_AMT_TOTAL_RECEIVABLE - AMT_RECIVABLE'] = credit['AMT_TOTAL_RECEIVABLE'] - credit['AMT_RECIVABLE']
credit['credit_card_AMT_TOTAL_RECEIVABLE - AMT_RECEIVABLE_PRINCIPAL'] = credit['AMT_TOTAL_RECEIVABLE'] - credit['AMT_RECEIVABLE_PRINCIPAL']
credit['credit_card_AMT_RECIVABLE - AMT_RECEIVABLE_PRINCIPAL'] = credit['AMT_RECIVABLE'] - credit['AMT_RECEIVABLE_PRINCIPAL']

credit['credit_card_AMT_BALANCE - AMT_RECIVABLE'] = credit['AMT_BALANCE'] - credit['AMT_RECIVABLE']
credit['credit_card_AMT_BALANCE - AMT_RECEIVABLE_PRINCIPAL'] = credit['AMT_BALANCE'] - credit['AMT_RECEIVABLE_PRINCIPAL']
credit['credit_card_AMT_BALANCE - AMT_TOTAL_RECEIVABLE'] = credit['AMT_BALANCE'] - credit['AMT_TOTAL_RECEIVABLE']
credit['credit_card_AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_ATM_CURRENT'] = credit['AMT_DRAWINGS_CURRENT'] - credit['AMT_DRAWINGS_ATM_CURRENT']
credit['credit_card_AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_OTHER_CURRENT'] = credit['AMT_DRAWINGS_CURRENT'] - credit['AMT_DRAWINGS_OTHER_CURRENT']
credit['credit_card_AMT_DRAWINGS_CURRENT - AMT_DRAWINGS_POS_CURRENT'] = credit['AMT_DRAWINGS_CURRENT'] - credit['AMT_DRAWINGS_POS_CURRENT']

credit_by_client = aggregate_client(credit, group_vars=['SK_ID_PREV', 'SK_ID_CURR'], df_names=['credit','client'])
credit_by_client.head()

train = train.merge(credit_by_client, on = 'SK_ID_CURR', how ='left')
test = test.merge(credit_by_client, on = 'SK_ID_CURR', how ='left')

gc.enable()
del credit, credit_by_client
gc.collect()
train, test = remove_missing_columns(train, test)

train.to_csv('train_after_stage2_1.csv', index = False) # 2600 features
test.to_csv('train_after_stage2_1.csv', index = False)

########### 6_ Installment Payments Data Set ##############################################################################

installments =pd.read_csv('installments_payments.csv')
installments = convert_types(installments, print_info = True)
installments.info()
installments.head()
# Replace some outliers
# Replace some outliers
installments.loc[installments['NUM_INSTALMENT_VERSION'] > 70, 'NUM_INSTALMENT_VERSION'] = np.nan
installments.loc[installments['DAYS_ENTRY_PAYMENT'] < -4000, 'DAYS_ENTRY_PAYMENT'] = np.nan


# Percentage and difference paid in each installment (amount paid and installment value)
installments['ins_PAYMENT_PERC'] = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT']
installments['ins_PAYMENT_DIFF'] = installments['AMT_INSTALMENT'] - installments['AMT_PAYMENT']
# Days past due and days before due (no negative values)
installments['ins_DPD'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
installments['ins_DBD'] = installments['DAYS_INSTALMENT'] - installments['DAYS_ENTRY_PAYMENT']
installments['ins_DPD'] = installments['DPD'].apply(lambda x: x if x > 0 else 0)
installments['ins_DBD'] = installments['DBD'].apply(lambda x: x if x > 0 else 0)
# Others
installments['ins_DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
installments['ins_NUM_INSTALMENT_NUMBER_100'] = (installments['NUM_INSTALMENT_NUMBER'] == 100).astype(int)
installments['ins_DAYS_INSTALMENT more NUM_INSTALMENT_NUMBER'] = (installments['DAYS_INSTALMENT'] > installments['NUM_INSTALMENT_NUMBER'] * 50 / 3 - 11500 / 3).astype(int)


installments_by_client = aggregate_client(installments, group_vars =['SK_ID_PREV', 'SK_ID_CURR'], df_names = ['installments', 'client'])

installments_by_client.head()

train=train.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left' )
test=test.merge(installments_by_client, on = 'SK_ID_CURR', how = 'left' )

gc.enable()
del installments, installments_by_client
gc.collect()
train, test = remove_missing_columns(train, test)
train.info()

print(f'Final training size: {return_size(train)}')


train.to_csv('train_after_stage3_1.csv', index = False) # 3200 features
test.to_csv('test_after_stage3_1.csv', index = False)

# Reference to write a function

def add_ratios_features(df):
    # CREDIT TO INCOME RATIO
    df['BUREAU_INCOME_CREDIT_RATIO'] = df['BUREAU_AMT_CREDIT_SUM_MEAN'] / df['AMT_INCOME_TOTAL']
    df['BUREAU_ACTIVE_CREDIT_TO_INCOME_RATIO'] = df['BUREAU_ACTIVE_AMT_CREDIT_SUM_SUM'] / df['AMT_INCOME_TOTAL']
    # PREVIOUS TO CURRENT CREDIT RATIO
    df['CURRENT_TO_APPROVED_CREDIT_MIN_RATIO'] = df['APPROVED_AMT_CREDIT_MIN'] / df['AMT_CREDIT']
    df['CURRENT_TO_APPROVED_CREDIT_MAX_RATIO'] = df['APPROVED_AMT_CREDIT_MAX'] / df['AMT_CREDIT']
    df['CURRENT_TO_APPROVED_CREDIT_MEAN_RATIO'] = df['APPROVED_AMT_CREDIT_MEAN'] / df['AMT_CREDIT']
    # PREVIOUS TO CURRENT ANNUITY RATIO
    df['CURRENT_TO_APPROVED_ANNUITY_MAX_RATIO'] = df['APPROVED_AMT_ANNUITY_MAX'] / df['AMT_ANNUITY']
    df['CURRENT_TO_APPROVED_ANNUITY_MEAN_RATIO'] = df['APPROVED_AMT_ANNUITY_MEAN'] / df['AMT_ANNUITY']
    df['PAYMENT_MIN_TO_ANNUITY_RATIO'] = df['INS_AMT_PAYMENT_MIN'] / df['AMT_ANNUITY']
    df['PAYMENT_MAX_TO_ANNUITY_RATIO'] = df['INS_AMT_PAYMENT_MAX'] / df['AMT_ANNUITY']
    df['PAYMENT_MEAN_TO_ANNUITY_RATIO'] = df['INS_AMT_PAYMENT_MEAN'] / df['AMT_ANNUITY']
    # PREVIOUS TO CURRENT CREDIT TO ANNUITY RATIO
    df['CTA_CREDIT_TO_ANNUITY_MAX_RATIO'] = df['APPROVED_CREDIT_TO_ANNUITY_RATIO_MAX'] / df[
        'CREDIT_TO_ANNUITY_RATIO']
    df['CTA_CREDIT_TO_ANNUITY_MEAN_RATIO'] = df['APPROVED_CREDIT_TO_ANNUITY_RATIO_MEAN'] / df[
        'CREDIT_TO_ANNUITY_RATIO']
    # DAYS DIFFERENCES AND RATIOS
    df['DAYS_DECISION_MEAN_TO_BIRTH'] = df['APPROVED_DAYS_DECISION_MEAN'] / df['DAYS_BIRTH']
    df['DAYS_CREDIT_MEAN_TO_BIRTH'] = df['BUREAU_DAYS_CREDIT_MEAN'] / df['DAYS_BIRTH']
    df['DAYS_DECISION_MEAN_TO_EMPLOYED'] = df['APPROVED_DAYS_DECISION_MEAN'] / df['DAYS_EMPLOYED']
    df['DAYS_CREDIT_MEAN_TO_EMPLOYED'] = df['BUREAU_DAYS_CREDIT_MEAN'] / df['DAYS_EMPLOYED']
    return df
# https://www.kaggle.com/jsaguiar/lightgbm-7th-place-solution




