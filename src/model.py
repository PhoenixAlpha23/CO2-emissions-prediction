from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib
from .data_preprocessing import create_preprocessor

def create_model_pipeline():
    """
    Create the full model pipeline including preprocessor and regressor.
    
    Returns:
        Pipeline: Sklearn pipeline with preprocessor and RandomForestRegressor
    """
    return Pipeline([
        ('preprocessor', create_preprocessor()),
        ('regressor', RandomForestRegressor())
    ])

def get_param_grid():
    """
    Define the parameter grid for hyperparameter tuning.
    
    Returns:
        dict: Parameter grid for GridSearchCV
    """
    return {
        'regressor__n_estimators': [100, 200, 300],
        'regressor__max_depth': [None, 5, 10],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }

def train_model(X, y):
    """
    Train the model using GridSearchCV for hyperparameter tuning.
    
    Args:
        X: Training features
        y: Target variable
    
    Returns:
        tuple: (best_model, best_parameters)
    """
    pipe = create_model_pipeline()
    param_grid = get_param_grid()
    
    grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=5)
    grid_search.fit(X, y)
    
    return grid_search.best_estimator_, grid_search.best_params_

def save_model(model, filepath):
    """
    Save the trained model to disk.
    
    Args:
        model: Trained model
        filepath (str): Path where to save the model
    """
    joblib.dump(model, filepath)

def load_model(filepath):
    """
    Load a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
    
    Returns:
        object: Loaded model
    """
    return joblib.load(filepath)
