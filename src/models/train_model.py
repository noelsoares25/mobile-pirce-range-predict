from src.features.feature_selection import get_features
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def find_best_model_with_params(X_train, y_train, X_test, y_test):
    
    models = { "RandomForestClassifier": RandomForestClassifier()        
    }
    params = {
        "RandomForestClassifier": {
            "n_estimators": [10, 15, 20, 50, 100],
            "max_depth": [6, 8, 10, 15, 20],
            "criterion": ['entropy', 'gini'],
            "max_features":  ["sqrt", "log2", None]
        }
    }
    
    def evaluate_model(models,params):
        for model_key,model in models.items():
            # print(model_key, model)
            param=params[model_key]
            rand=RandomizedSearchCV(model,param,cv=5,n_jobs=-1,scoring='accuracy')
            rand.fit(X_train,y_train)
            y_pred=rand.predict(X_test)
            best_p = rand.best_params_
            acc = accuracy_score(y_test,y_pred)
            
        return model, best_p, acc
            
    model, best_p, acc = evaluate_model(models,params)
    
    return model, best_p, acc

def main():
    X,y = get_features()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model, best_p, acc = find_best_model_with_params(X_train, y_train, X_test, y_test)
    print(model, best_p, acc)
    # save_model(trained_model, output_path)   
    # We will push this model to S3 and also copy in the root folder for Dockerfile to pick


if __name__ == "__main__":
    main()