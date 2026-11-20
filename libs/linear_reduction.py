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