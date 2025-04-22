import sklearn
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import joblib
import json

def main():
    df = pd.read_csv('dataset.csv').drop(columns=['Unnamed: 0', 'actual_balance'])
    num_col = df.select_dtypes(include = [np.number]).columns.values
    neg_col = [c for c in num_col if df[df[c] < 0][c].count() > 0]

    for col in neg_col:
        mask = df[col] < 0
        df.loc[mask, col] = 0

    #imputer = SimpleImputer(strategy = 'median')
    #df_imp = pd.DataFrame(imputer.fit_transform(df.drop(columns = ['address','label'])), columns = df.drop(columns = ['address','label']).columns)

    volume_cols = ['n_transactions', 
               'n_in_tx', 
               'n_out_tx', 
               'Max_daily_tx', 
               'n_reused_address_in', 
               'n_reused_address_out', 
               'min_daily_tx',
               'Max_daily_tx',
               'avg_daily_tx',
               'Max_in_tx_size', 
               'min_in_tx_size', 
               'avg_in_tx_size', 
               'Max_out_tx_size', 
               'min_out_tx_size', 
               'avg_out_tx_size']
    eps = 1
    for col in volume_cols:
        df[col] = np.log(df[col] + eps)
    
    X = df.drop(columns = ['address','label'])
    Y = df['label']
    pipeline = Pipeline([
        ('standard_scaler', StandardScaler(with_std=True)),
        ('minmax_scaler', MinMaxScaler(feature_range = (0,1)))
    ])
    
    X_scaled = pipeline.fit_transform(X)

    #RANDOM FOREST
    if not os.path.isfile('best_rfc.joblib'):
        
        params = {
            'n_estimators': [500, 700, 1000],
            'max_depth': [50, 100, 200],
            'min_samples_split': [5],
            'min_samples_leaf': [1],
            'max_features': ['sqrt'],
            'bootstrap': [False]
        }
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
        rfc = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(rfc, params, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        
        model = grid_search.best_estimator_

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        fold_accuracies = []
        best_score = 0
        best_model = None
        print(f"########### RANDOM FOREST ###########")
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            fold_accuracies.append(f1)
            if f1 > best_score:
                best_score = f1
                best_model = model
        
        joblib.dump(best_model, 'best_rfc.joblib')
        print(f"Fold F1_score: {fold_accuracies}")
        print(f"Mean Accuracy: {np.mean(fold_accuracies)}")
        print(f"\n####################################\n\n")

    #GRADIENT BOOSTING
    if not os.path.isfile('best_gbc.joblib'):

        print(f"############# GRADIENT BOOST #############")
        if not os.path.isfile('best_gb_params.json'):
            print(f"Grid Search...")    
            params = {
                'learning_rate': [0.5],
                'max_depth': [10, 30],
                'n_estimators': [200, 500],
                'max_features': ["sqrt"],
                'subsample': [1],
                'min_samples_leaf': [0.02, 0.1],
                'min_samples_split': [0.2],
                'min_weight_fraction_leaf': [0.0],
                'min_impurity_decrease': [0.0],
            }

            grid_search = GridSearchCV(GradientBoostingClassifier(), params, cv=5, scoring='f1_weighted')
            grid_search.fit(X_scaled, Y)
            params = grid_search.best_params_
            print(f"Best params: {params}")
            with open('best_gb_params.json', 'w') as f:
                json.dump(params, f)
        else:
            with open('best_gb_params.json', 'r') as f:
                params = json.load(f)
        print('K-Fold...')
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        model = grid_search.best_estimator_
        fold_accuracies = []
        best_score = 0
        best_model = None
        	
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            fold_accuracies.append(f1)
            if f1 > best_score:
                best_score = f1
                best_model = model
                print(classification_report(y_test, y_pred))

        joblib.dump(best_model, 'best_gbc.joblib')
        
        print(f"Fold F1_score: {fold_accuracies}")
        print(f"Mean Accuracy: {np.mean(fold_accuracies)}")
        print(f"\n####################################\n\n")

    
    #K-NN
    if not os.path.isfile('best_knn.joblib'):
        best_neighs = 0
        best_score = 0
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)

        # Find best N
        for i in range(1,50):
            model = KNeighborsClassifier(n_neighbors=i)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            if f1 > best_score:
                best_score = f1
                best_neighs = i

        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        model = KNeighborsClassifier(n_neighbors=best_neighs)
        fold_accuracies = []
        best_score = 0
        best_model = None
        print(f"################# K-NN ################")
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            fold_accuracies.append(f1)
            if f1 > best_score:
                best_score = f1
                best_model = model

        joblib.dump(best_model, 'best_knn.joblib')
        
        print(f"Fold Accuracies: {fold_accuracies}")
        print(f"Mean Accuracy: {np.mean(fold_accuracies)}")
        print(f"\n####################################\n\n")      

    #EXTRA-TREES
    if not os.path.isfile('best_extree.joblib'):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.3, random_state=42)
        params = {
            'n_estimators': [700, 1000],
            'max_depth': [50],
            'min_samples_split': [5],
            'min_samples_leaf': [1],
            'max_features': ['sqrt'],
            'bootstrap': [False]
        }
        ext = ExtraTreesClassifier(random_state=42)
        grid_search = GridSearchCV(ext, params, cv=5, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)
        
        model = grid_search.best_estimator_
        best_score = 0
        

        fold_accuracies = []
        best_score = 0
        best_model = None
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        print(f"################# EXTRA TREES ################")
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = f1_score(y_test, y_pred, average='weighted')
            fold_accuracies.append(f1)
            if f1 > best_score:
                best_score = f1
                best_model = model
                print(classification_report(y_test, y_pred))
        
        joblib.dump(best_model, 'best_extree.joblib')
        
        print(f"Fold Accuracies: {fold_accuracies}")
        print(f"Mean Accuracy: {np.mean(fold_accuracies)}")
        print(f"\n####################################\n\n")

    #XGBOOST
    if not os.path.isfile('best_xgb.joblib'):
        le = LabelEncoder()
        Y_n = le.fit_transform(Y)

        print(f"################# XGBOOST ################")
        if not os.path.isfile('best_xgb_params.json'):
            print(f"Grid Search...")
            params = {
            'objective': ['multi:softmax'],
            'num_class': [7],
            'learning_rate': [0.07, 0.3, 0.5],
            'max_depth': [6, 8, 10],
            'gamma': [0, 0.05, 0.5],
            'subsample': [0.5, 0.75, 1],
            'colsample_bytree': [0.5, 0.8, 1],
            'min_child_weight': [0, 0.5, 5, 10],
            'reg_lambda': [0, 0.05, 0.5, 5, 10],
            'reg_alpha': [0.05, 0.5, 1, 5]
            }
            #BEST PARAMS:
            # Params: {'colsample_bytree': 0.8, 
            #          'gamma': 0, 
            #          'learning_rate': 0.3, 
            #          'max_depth': 8,
            #          'min_child_weight': 0,
            #          'reg_alpha': 1, 
            #          'reg_lambda': 0.05,
            #          'num_class': 7,
            #          'objective': 'multi:softmax', 
            #          'subsample': 1}
            xgb_model = xgb.XGBClassifier(eval_metric="mlogloss")
            grid_search = GridSearchCV(estimator=xgb_model, param_grid=params, cv=5, scoring='f1_weighted')
            grid_search.fit(X_scaled, Y_n)
            params = grid_search.best_params_
            print(params)
            with open('best_xgb_params.json', 'w') as f:
                json.dump(params, f)
        else:
            with open('best_xgb_params.json', 'r') as f:
                params = json.load(f)

        fold_accuracies = []
        best_score = 0
        best_model = None
        print(f"K-Fold...")
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X_scaled):
            X_train, X_test = X_scaled[train_index], X_scaled[test_index]
            y_train, y_test = Y_n[train_index], Y_n[test_index]
         
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            model = xgb.train(params, dtrain, num_boost_round=100)

            y_pred = model.predict(dtest)
            f1 = f1_score(y_test, y_pred, average='weighted')
            fold_accuracies.append(f1)
            if f1 > best_score:
                best_score = f1
                best_model = model
                print(classification_report(y_test, y_pred.astype(int)))

        joblib.dump(best_model, 'best_xgb.joblib')
        print(f"Params: {params}")
        print(f"Fold Accuracies: {fold_accuracies}")
        print(f"Mean Accuracy: {np.mean(fold_accuracies)}")
        print(f"\n####################################\n\n")

    

if __name__ == "__main__":
    main()


