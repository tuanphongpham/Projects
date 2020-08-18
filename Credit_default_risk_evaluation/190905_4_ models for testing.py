# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 23:54:01 2019

@author: Phong
"""

############## Model function for testing  ('logistic_model', 'random_forest_model', 'xg_boost_model', 'lgb_model') using Cross Validation
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb


def model(features, test_features, encoding = 'ohe', n_folds = 5, im ='median', used_model = 'lgb_model'):
   
    """Train and test a model ('logistic_model', 'random_forest_model', 'xg_boost_model', 'lgb_model') using cross validation. 
    
    Parameters
    --------
        features (pd.DataFrame): 
            dataframe of training features to use 
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame): 
            dataframe of testing features to use
            for making predictions with the model. 
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): 
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame): 
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame): 
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
    """
    # Extract the ids:  features = train , test_features= test
    features, test_features = remove_missing_columns(features, test_features, threshold = 99.9)
    train_ids = features ['SK_ID_CURR'] 
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    train.shape
    test.shape
    features.shape
    test_features.shape
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    # One Hot Encoding
    
    if encoding =='ohe':
        features =pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Align the data by columns:
        features, test_features = features.align(test_features, join='inner', axis = 1)
        # No categorical indices to record
        cat_indices = 'auto'
     
    # Integer Label Encoding
    elif encoding == 'le':
        # Create a label encoder
        label_encoder = LabelEncoder()
        # List for storing categorical indices
        cat_indices = []
        # Iterate through each column:
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to intergers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_featutes).astype(str).reshape((-1,)))
                
                # Record the categorical indices
                cat_indices.append(i)
                
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # Extract feature names
    feature_names = list(features.columns)
    ##### Median imputation of missing values for 'logistic_model'|'random_forest_model'
    if used_model == 'logistic_model'or'random_forest_model':
        
        if im=='median':
        
            imputer = Imputer(strategy ='median') # can replaced by 'most_frequent' or 'constant'

            imputer.fit(features)
            features = imputer.transform(features)
            test_features = imputer.transform(test_features)
        elif im =='zero':
            features.fillna(0, inplace=True)
            test_features.fillna(0, inplace=True)
        elif im =='number':
            features.fillna(30000, inplace=True)
            test_features.fillna(30000, inplace=True)
        else:
            print("NA must be filled by 'median' or '0' or 'a number'")
        

    ##### Scale to 0-1 for logistic regression
    if used_model == 'logistic_model':
        scaler = MinMaxScaler(feature_range =(0,1))
        scaler.fit(features)
        features = scaler.transform(features)
        test_features = scaler.transform(test_features)
   
    
    # Convert to np arrays
    if used_model == 'xg_boost_model'or'lgb_model':
        features = np.array(features)
        test_features = np.array(test_features)
      
       
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = False, random_state = 50)
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    # Empty array for out of fold validation prediction
    out_of_fold = np.zeros(features.shape[0])
    train_prediction = np.zeros(features.shape[0])
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    if used_model == 'logistic_model':
        for train_indices, valid_indices in k_fold.split(features):
            print(train_indices, valid_indices)
            train_features, train_labels = features[train_indices], labels[train_indices]
        
            valid_features, valid_labels = features[valid_indices], labels[valid_indices]
            # Create the model
        
            model = LogisticRegression(C=0.0001)
            # Train the model
            model.fit(train_features, train_labels)
                    
            # Make predictions
            train_prediction = model.predict_proba(train_features)[:,1]
            train_auc = roc_auc_score(train_labels, train_prediction)
            train_scores.append(train_auc)
            valid_prediction = model.predict_proba(valid_features)[:,1]
            valid_auc = roc_auc_score(valid_labels, valid_prediction)
            valid_scores.append(valid_auc)
            test_predictions += model.predict_proba(test_features)[:,1]/k_fold.n_splits
        
            # Record the out of fold predictions
            out_of_fold[valid_indices] = model.predict_proba(valid_features)[:,1]
        
            # Clean up memory
            gc.enable()
            del model, train_features, valid_features
            gc.collect()
        
        # Overall validation score
        valid_auc = roc_auc_score(labels, out_of_fold)
        # Add the overall scores to the metrics
        valid_scores.append(valid_auc)
        train_scores.append(np.mean(train_scores))
    
        # creating dataframe of validation scores
        fold_names = list(range(n_folds))
        fold_names.append('overall')
    
        # Dataframe of validation scores
        metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})
        return metrics, test_predictions
    
    elif used_model == 'random_forest_model':
        for train_indices, valid_indices in k_fold.split(features):
            train_features, train_labels = features[train_indices], labels[train_indices]
            valid_features, valid_labels = features[valid_indices], labels[valid_indices]
            # Create the model
        
            model = RandomForestClassifier(n_estimators = 1000, random_state = 50, verbose = 1, n_jobs = -1, max_depth= 10)
            # Train the model
            model.fit(train_features, train_labels)
            #model.fit(features, labels)
        
            # Record the feature importances
            feature_importance_values += model.feature_importances_/k_fold.n_splits 
                
            # Make predictions
            train_prediction = model.predict_proba(train_features)[:,1]
            train_auc = roc_auc_score(train_labels, train_prediction)
            train_scores.append(train_auc)
            valid_prediction = model.predict_proba(valid_features)[:,1]
            valid_auc = roc_auc_score(valid_labels, valid_prediction)
            valid_scores.append(valid_auc)
            test_predictions += model.predict_proba(test_features)[:,1]/k_fold.n_splits
        
            # Record the out of fold predictions
            out_of_fold[valid_indices] = model.predict_proba(valid_features)[:,1]
        
            # Clean up memory
            gc.enable()
            del model, train_features, valid_features
            gc.collect()
        
        # Overall validation score
        valid_auc = roc_auc_score(labels, out_of_fold)
        # Add the overall scores to the metrics
        valid_scores.append(valid_auc)
        train_scores.append(np.mean(train_scores))
    
        # creating dataframe of validation scores
        fold_names = list(range(n_folds))
        fold_names.append('overall')
    
    
        # Dataframe of validation scores
        metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})
    
        # Make the feature importance dataframe
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values })
        feature_importances = feature_importances.sort_values('importance', ascending = False)
        return metrics, feature_importances, test_predictions
    elif used_model == 'xg_boost_model':
        for train_indices, valid_indices in k_fold.split(features):
            train_features, train_labels = features[train_indices], labels[train_indices]
            valid_features, valid_labels = features[valid_indices], labels[valid_indices]
            # Create the model
        
            params = {'objective': 'binary:logistic',
                      'max_depth': 5,
                      'learning_rate': 0.05,# 0.005
                      'silent': False,
                      'n_estimators': 5000,
                      'n_jobs=':-1
                  }
            
            #params = {'objective': 'binary:logistic',
            #          'max_depth': 5,
            #          'learning_rate': 0.01,# 0.005
            #          'silent': False,
            #          'n_estimators': 5000,
            #          "gamma": 0.0, 
            #          "min_child_weight": 10, # default: 1
            #          "subsample": 0.7, 
            #          "colsample_bytree": 0.7,  # default:  1.0
            #          "colsample_bylevel": 0.5, # default: 1.0
            #          "reg_alpha": 0.0, 
            #          "reg_lambda": 1.0, 
            #          "scale_pos_weight": 1.0, 
            #          "random_state": 0,
            ##          #
            #          "silent": False, 
            #          "n_jobs": 16, 
            #          #
            #          "tree_method": "gpu_hist", # default: auto
            #          "grow_policy": "lossguide", # default depthwise
            #          "max_leaves": 0, # default: 0(unlimited)
            #          "max_bin": 256  # default: 256
            #          }
        
            model = XGBClassifier(**params)
            # Train the model
        
            model.fit(train_features, train_labels,eval_set = [(train_features, train_labels), (valid_features, valid_labels)],
                                                               eval_metric = 'auc', early_stopping_rounds = 100, verbose=True)
            # record the best iteration
        
            best_iteration = model.best_iteration
                       
            # Record the feature importances
            feature_importance_values += model.feature_importances_/k_fold.n_splits 
                
            # Make predictions
            train_prediction = model.predict_proba(train_features)[:,1]
            train_auc = roc_auc_score(train_labels, train_prediction)
            train_scores.append(train_auc)
            valid_prediction = model.predict_proba(valid_features)[:,1]
            valid_auc = roc_auc_score(valid_labels, valid_prediction)
            valid_scores.append(valid_auc)
            test_predictions += model.predict_proba(test_features, ntree_limit = best_iteration)[:,1]/k_fold.n_splits
        
            # Record the out of fold predictions
            out_of_fold[valid_indices] = model.predict_proba(valid_features, ntree_limit = best_iteration)[:,1]
        
            # Clean up memory
            gc.enable()
            del model, train_features, valid_features
            gc.collect()
        
        # Overall validation score
        valid_auc = roc_auc_score(labels, out_of_fold)
        # Add the overall scores to the metrics
        valid_scores.append(valid_auc)
        train_scores.append(np.mean(train_scores))
    
        # creating dataframe of validation scores
        fold_names = list(range(n_folds))
        fold_names.append('overall')
    
    
        # Dataframe of validation scores
        metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores})
    
        # Make the feature importance dataframe
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values })
        feature_importances = feature_importances.sort_values('importance', ascending = False)
        return metrics, feature_importances, test_predictions
    else:
        for train_indices, valid_indices in k_fold.split(features):
        
            # Training data for the fold
            train_features, train_labels = features[train_indices], labels[train_indices]
            # Validation data for the fold
            valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
            # Create the model
            model = lgb.LGBMClassifier(n_estimators=10000, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.05, 
                                   reg_alpha = 0.1, reg_lambda = 0.1, 
                                   subsample = 0.8, n_jobs = -1, random_state = 50)
        
            # Train the model
            model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 100, verbose = 200)
        
            # Record the best iteration
            best_iteration = model.best_iteration_
        
            # Record the feature importances
            feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
            # Make predictions
            test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
            # Record the out of fold predictions
            out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
            # Record the best score
            valid_score = model.best_score_['valid']['auc']
            train_score = model.best_score_['train']['auc']
        
            valid_scores.append(valid_score)
            train_scores.append(train_score)
        
            # Clean up memory
            gc.enable()
            del model, train_features, valid_features
            gc.collect()
        
        # Make the submission dataframe
        submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
        # Make the feature importance dataframe
        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
        # Overall validation score
        valid_auc = roc_auc_score(labels, out_of_fold)
        # test_auc = roc_auc_score(test_labels, test_predictions)
    
        # Add the overall scores to the metrics
        valid_scores.append(valid_auc)
        train_scores.append(np.mean(train_scores))
    
        # Needed for creating dataframe of validation scores
        fold_names = list(range(n_folds))
        fold_names.append('overall')
    
        # Dataframe of validation scores
        metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
        return metrics, feature_importances, test_predictions

        
        
 