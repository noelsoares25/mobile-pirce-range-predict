import pathlib
import sys
import joblib
import mlflow

from src.features.feature_selection import get_features
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def find_best_model_with_params(X_train, y_train, X_test, y_test):
    
    models = { "RandomForestClassifier": RandomForestClassifier()        
    }
    params = {
        "RandomForestClassifier": {
            "n_estimators": [10, 15, 20, 50, 100,150],
            "max_depth": [6, 8, 10, 15, 20, 50, 100, None],
            "criterion": ['entropy', 'gini'],
            "max_features":  ["sqrt", "log2", None]
        }
    }
    
    def evaluate_model(models,params):
        best_model = None
        best_params = None
        best_accuracy = 0
        
        for model_key,model in models.items():
            # print(model_key, model)
            param=params[model_key]
            rand=RandomizedSearchCV(model,param,cv=5,n_jobs=-1,scoring='accuracy')
            rand.fit(X_train,y_train)
            y_pred=rand.predict(X_test)
            best_p = rand.best_params_
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')            
            
            with mlflow.start_run():
                mlflow.sklearn.log_model(rand.best_estimator_, model_key)
                mlflow.log_param("model", model_key)
                mlflow.log_params(rand.best_params_)
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("precision", precision)
                mlflow.log_metric("recall", recall)
                mlflow.log_metric("f1_score", f1)
                
            # Update the best model if the current one is better
            if accuracy > best_accuracy:
                best_model = rand.best_estimator_
                best_params = rand.best_params_
                best_accuracy = accuracy
        
        return best_model, best_params, best_accuracy
    
    
            
    model, best_p, acc = evaluate_model(models,params)
    
    return model, best_p, acc


def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + "/model.joblib")

def main():
    
    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    output_path = home_dir.as_posix() + "/models"
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    
    X,y = get_features()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    trained_model, best_p, accuracy = find_best_model_with_params(X_train, y_train, X_test, y_test)
    print(trained_model, best_p, accuracy)
    save_model(trained_model, output_path)   
    # We will push this model to S3 and also copy in the root folder for Dockerfile to pick


if __name__ == "__main__":
    main()