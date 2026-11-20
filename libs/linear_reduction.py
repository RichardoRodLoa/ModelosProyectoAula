import pandas as pd 
from sklearn.metrics import accuracy_score, f1_score, recall_score

def train_models(xtrain, xtest, ytrain, ytest, models, params):
    rows = []
    for model in models: 
        
        model.fit(xtrain, ytrain)
        
        predictions_train = model.predict(xtrain)
        predictsion_test = model.predict(xtest)
        
    
        rows.append(
            dict([(param, model.get_params()[param]) for param in params]) | 
            {
            'acc train': accuracy_score(ytrain, predictions_train), 
            'acc test': accuracy_score(ytest, predictsion_test),
            'f1_score': f1_score(ytest, predictsion_test), 
            'recall': recall_score(ytest, predictsion_test) 
            }
        )

    return pd.DataFrame(rows)

import umap
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score
import numpy as np
from sklearn.base import clone 
import pandas as pd 

def validate_umap_reduction(X, y, model): 
    df = pd.DataFrame(np.asarray(X)); df["y"] = np.asarray(y)
    df = df.drop_duplicates()
    X, y = df.drop(columns="y").values, df["y"].values

    components = [2, 5, 10, 20]
    neighbors = [10, 15, 30]
    min_dists = [0.0, 0.1]

    rows = []
    for c in components: 
        for n in neighbors: 
            for d in min_dists: 
                
                accs_train, accs_test, f1s, recalls = [], [], [], []
                
                kf = KFold(n_splits=5)
                for index_train, index_test in kf.split(X): 
                    xtrain, xtest = X[index_train], X[index_test]
                    ytrain, ytest = y[index_train], y[index_test]
                    
                    scaler = StandardScaler()
                    xtrain, xtest = scaler.fit_transform(xtrain), scaler.transform(xtest)
                
                    reducer = umap.UMAP(n_components=c, n_neighbors=n, min_dist=d)
                    xtrain, xtest = reducer.fit_transform(xtrain), reducer.transform(xtest)
                    
                    m = clone(model)
                    m.fit(xtrain, ytrain)
                    
                    prediction_train = m.predict(xtrain)
                    prediction_test = m.predict(xtest)
                    
                    accs_train.append(accuracy_score(ytrain, prediction_train))
                    accs_test.append(accuracy_score(ytest, prediction_test))
                    f1s.append(f1_score(ytest, prediction_test))
                    recalls.append(recall_score(ytest, prediction_test))
            
                rows.append({
                    'n_components': c, 
                    'n_neighbors': n, 
                    'min_dist': d, 
                    'mean_train_acc': np.mean(accs_train), 
                    'mean_test_acc': np.mean(accs_test), 
                    'ic_acc': np.std(accs_test), 
                    'mean_f1': np.mean(f1s), 
                    'ic_f1': np.std(f1s), 
                    'mean_recall': np.mean(recalls), 
                    'ic_recall': np.std(recalls)
                })
            
    return pd.DataFrame(rows).sort_values(by='mean_f1', ascending=False)
