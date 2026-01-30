
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

import kagglehub

def download_dataset(dataset_slug):
    """
    Download dataset from Kaggle using kagglehub.
    """
    try:
        path = kagglehub.dataset_download(dataset_slug)
        # Find the csv file in the path
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".csv"):
                    return os.path.join(root, file)
    except Exception as e:
        print(f"Error downloading dataset {dataset_slug}: {e}")
        return None
    return None

def load_data():
    """
    Load data.

    """
    # Check for local dataset
    local_path = os.path.join("data", "student_performance.csv")
    if os.path.exists(local_path):
        print(f"Loading local dataset from {local_path}...")
        df = pd.read_csv(local_path)
    else:
        print("Local dataset not found. Attempting to download from Kaggle...")
        # Download from Kaggle
        # Dataset 1: Student Performance Factors
        path1 = download_dataset("ayeshaseherr/student-performance")
        # Dataset 2: Social Media Addiction vs Relationships
        path2 = download_dataset("adilshamim8/social-media-addiction-vs-relationships")

        if not path1 or not path2:
            raise FileNotFoundError("Could not find local dataset and failed to download required datasets from Kaggle.")

        # Load datasets
        df_academic = pd.read_csv(path1)
        df_social = pd.read_csv(path2)

        # Clean column names (strip spaces)
        df_academic.columns = df_academic.columns.str.strip()
        df_social.columns = df_social.columns.str.strip()

        # Merge datasets (index-based inner join)
        df = pd.merge(df_academic, df_social, left_index=True, right_index=True, suffixes=('', '_social'))
        
        # Save merged version locally for next time
        os.makedirs("data", exist_ok=True)
        df.to_csv(local_path, index=False)
        print(f"Dataset saved to {local_path}")

    # Drop unnamed index column if it exists
    if not df.empty and df.columns[0].startswith('Unnamed'):
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
