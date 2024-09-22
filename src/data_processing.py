import pandas as pd
import os

def get_project_root():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return project_root

def load_data():
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'diabetes_prediction_dataset.csv')
    df = pd.read_csv(data_path)
    return df

def preprocess_data(df):
    # Dropping the empty and none values
    df = df.dropna()

    # Dropping duplicates
    df.drop_duplicates(inplace=True)

    # Checking info on columns
    #print(df['smoking_history'].value_counts())

    # One hot encoding
    df = pd.get_dummies(df, columns=['smoking_history', 'gender'],
                        drop_first=True)

    # Print the dataset
    #pd.set_option('display.max_columns', None)
    #print(df.head())

    # Return df
    return df