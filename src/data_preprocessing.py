import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(file_path):
    """
    Load and prepare the CO2 emissions dataset.
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Processed dataframe with selected features
    """
    df = pd.read_csv(file_path)
    
    extra_columns = ['Make', 'Model', 'Cylinders', 'Transmission', 
                    'Fuel Consumption City (L/100 km)',
                    'Fuel Consumption Hwy (L/100 km)', 
                    'Fuel Consumption Comb (mpg)']
    
    data = df.drop(extra_columns, axis=1)
    return data

def create_preprocessor():
    """
    Create a preprocessor pipeline for numerical and categorical features.
    
    Returns:
        ColumnTransformer: Preprocessor pipeline
    """
    categorical_features = ['Vehicle Class', 'Fuel Type']
    numerical_features = ['Engine Size(L)', 'Fuel Consumption Comb (L/100 km)']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), 
             categorical_features)
        ])
    
    return preprocessor

def prepare_data(data, target_column='CO2 Emissions(g/km)', test_size=0.2):
    """
    Split data into features and target, then into train and test sets.
    
    Args:
        data (pd.DataFrame): Input dataframe
        target_column (str): Name of the target column
        test_size (float): Proportion of dataset to include in the test split
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    return train_test_split(X, y, test_size=test_size)

def get_vehicle_type_distribution(data):
    """
    Get the distribution of vehicle types in the dataset.
    
    Args:
        data (pd.DataFrame): Input dataframe
    
    Returns:
        pd.Series: Count of each vehicle type
    """
    return data['Vehicle Class'].value_counts()
