
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data():
    """
    Load dataset.
    """
    paths = [
        os.path.join("data", "student_performance.csv"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "student_performance.csv")
    ]
    
    path = None
    for p in paths:
        if os.path.exists(p):
            path = p
            break
            
    if not path:
        raise FileNotFoundError(f"Could not find the dataset. Checked: {paths}")

    df = pd.read_csv(path)
    
    # Drop unnamed index column if it exists
    if df.columns[0].startswith('Unnamed'):
        df = df.drop(df.columns[0], axis=1)
        
    return clean_data(df)


def clean_data(df):
    """
    Clean the dataset by handling missing values and data types.
    """
    df = df.copy()
    
    # Handle missing values
    # For numerical columns, fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
            
    # Normalize column names
    df.columns = [c.replace(' ', '_') for c in df.columns]
    
    return df


def encode_categorical(df):
    """
    Encode categorical variables using Label Encoding.
    Returns the encoded dataframe and the encoders dictionary.
    """
    df_encoded = df.copy()
    encoders = {}
    
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    
    return df_encoded, encoders


def get_feature_target_split(df, target_col='Exam_Score'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_data_for_classification(y, threshold=60):
    return (y >= threshold).astype(int)


def get_data_summary(df):
    summary = {
        'n_samples': len(df),
        'n_features': len(df.columns) - 1,
        'numerical_features': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_features': list(df.select_dtypes(include=['object']).columns),
        'missing_values': df.isnull().sum().to_dict(),
        'target_stats': {
            'mean': df['Exam_Score'].mean() if 'Exam_Score' in df.columns else None,
            'std': df['Exam_Score'].std() if 'Exam_Score' in df.columns else None,
            'min': df['Exam_Score'].min() if 'Exam_Score' in df.columns else None,
            'max': df['Exam_Score'].max() if 'Exam_Score' in df.columns else None
        }
    }
    return summary
