# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 23:50:27 2019

@author: Phong
"""

######### PART 4: Test  Models with 'train_after_stage3_1.csv' data  ############
print('Read the data and eliminate infinity values and columns with missing values > 99%')

train = pd.read_csv('train_after_stage3_1_1.csv')
test = pd.read_csv('test_after_stage3_1_1.csv').drop(columns =['TARGET'])
test_labels =  pd.read_csv('test_after_stage3_1_1.csv')['TARGET']
train = train.replace([np.inf, -np.inf], np.nan)
test = test.replace([np.inf, -np.inf], np.nan)
train, test = remove_missing_columns(train, test, threshold = 99)


# Logistic Model
print('Start running Logistic Model for train_after_stage3_1.csv data ')
auc_lg3, prediction_lg3= model(train, test, used_model = 'logistic_model' )  
test_auc_lg3 = roc_auc_score(test_labels, prediction_lg3)
logic = ['No']*5
logic.append(test_auc_lg3)
auc_lg3['logic'] = logic
auc_lg3
# Random Forest Model
print('Start running Random Forest Model for train_after_stage3_1.csv data ')
auc_rf3, feature_importances_rf3, prediction_rf3 = model(train, test, used_model = 'random_forest_model') 
test_auc_rf3 = roc_auc_score(test_labels, prediction_rf3)
logic = ['No']*5
logic.append(test_auc_rf3)
auc_rf3['random_forest'] = logic
auc_rf3
# Light Gradient Boosting Model
print('Start running Light Gradient Boosting Model for train_after_stage3_1.csv data ')
auc_lgb3, feature_importances_lgb3, prediction_lgb3 = model(train, test, used_model = 'lgb_model')
test_auc_lgb3 = roc_auc_score(test_labels, prediction_lgb3)
logic = ['No']*5
logic.append(test_auc_lgb3)
auc_lgb3['light gradient boosting'] = logic
auc_lgb3
# XG boosting Model
print('Start running XG boosting Model for train_after_stage3_1.csv data ') 
auc_xg3, feature_importances_xg3, prediction_xg3 =  model(train, test, used_model = 'xg_boost_model')
test_auc_xg3 = roc_auc_score(test_labels, prediction_xg3)

logic = ['No']*5
logic.append(test_auc_xg3)
auc_xg3['xg_boosting'] = logic
auc_xg3

print('Summary results for train_after_stage3_1.csv data ')
testing_summary_stage3 = pd.concat([auc_lg3,auc_rf3, auc_xg3,auc_lgb3 ], axis=1) 
testing_summary_stage3

# Can try blen the models: https://www.kaggle.com/ishaan45/thank-you