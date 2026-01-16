"""
Data Loader Module
Handles loading, cleaning, and preprocessing the student performance dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def download_dataset(dataset_slug):
    """
    Download dataset from Kaggle using kagglehub.
    """
    import kagglehub
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
    Load data from the two specified Kaggle datasets and merge them.
    1. ayeshaseherr/student-performance (Academic focus)
    2. adilshamim8/social-media-addiction-vs-relationships (Social media focus)
    """
    # Dataset 1: Student Performance Factors
    path1 = download_dataset("ayeshaseherr/student-performance")
    if not path1:
        # Fallback to local if exists
        path1 = os.path.join("data", "student_performance.csv")
    
    # Dataset 2: Social Media Addiction vs Relationships
    path2 = download_dataset("adilshamim8/social-media-addiction-vs-relationships")
    if not path2:
        path2 = os.path.join("data", "social_media_addiction.csv")

    if not os.path.exists(path1) or not os.path.exists(path2):
        raise FileNotFoundError("Could not find the required datasets.")

    df_academic = pd.read_csv(path1)
    df_social = pd.read_csv(path2)

    # Clean column names (strip spaces, lower case)
    df_academic.columns = df_academic.columns.str.strip()
    df_social.columns = df_social.columns.str.strip()

    # Pre-merge cleaning and mapping for Academic Dataset
    # ayeshaseherr/student-performance columns:
    # Hours Studied, Attendance, Parental Involvement, Access to Resources, Extracurricular Activities, 
    # Sleep Hours, Previous Scores, Motivation Level, Internet Access, Tutoring Sessions, 
    # Family Income, Teacher Quality, School Type, Peer Influence, Physical Activity, 
    # Learning Disabilities, Parental Education Level, Distance from Home, Gender, Exam Score
    
    # Mapping for academic
    rename_academic = {
        'Hours Studied': 'Hours_Studied',
        'Attendance': 'Attendance',
        'Sleep Hours': 'Sleep_Hours',
        'Previous Scores': 'Previous_Scores',
        'Motivation Level': 'Motivation_Level',
        'Physical Activity': 'Physical_Activity',
        'Exam Score': 'Exam_Score',
        'Parental Education Level': 'Parental_Education_Level',
        'Extracurricular Activities': 'Extracurricular_Activities'
    }
    df_academic.rename(columns=rename_academic, inplace=True)

    # Mapping for social media dataset
    # adilshamim8/social-media-addiction-vs-relationships columns:
    # Student_ID, Age, Gender, Academic_Level, Country, Avg_Daily_Usage_Hours, Most_Used_Platform, 
    # Affects_Academic_Performance, Sleep_Hours_Per_Night, Mental_Health_Score, Relationship_Status, 
    # Conflicts_Over_Social_Media, Addicted_Score
    
    rename_social = {
        'Avg_Daily_Usage_Hours': 'Social_Media_Hours',
        'Mental_Health_Score': 'mental_health_rating',
        'Relationship_Status': 'Relationship_Status',
        'Addicted_Score': 'Social_Media_Addiction_Level',
        'Most_Used_Platform': 'Primary_Platform',
        'Gender': 'Social_Gender',
        'Age': 'Social_Age'
    }
    df_social.rename(columns=rename_social, inplace=True)

    # Since we can't join on ID, we'll join statistically or just merge row-wise 
    # if we want to combine features for a richer dataset.
    # To maintain consistency, let's try to match by Gender and Age if available in both.
    # df_academic doesn't have Age in some versions, let's check.
    # If not, we'll just sample from df_social to fill df_academic.
    
    # Let's ensure df_academic is the primary
    df = df_academic.copy()
    
    # Randomly assign social media data to academic data to create a combined "real-like" dataset
    # This is better than pure simulation because it preserves real correlated patterns 
    # from the social media dataset (e.g. mental health vs usage hours).
    
    # Sample enough rows from df_social to match df_academic
    if len(df_social) < len(df):
        # Upsample social media data
        df_social_expanded = df_social.sample(n=len(df), replace=True, random_state=42).reset_index(drop=True)
    else:
        df_social_expanded = df_social.sample(n=len(df), random_state=42).reset_index(drop=True)

    # Combine columns
    # We take real social media usage and health metrics from the social dataset
    df['Social_Media_Hours'] = df_social_expanded['Social_Media_Hours']
    df['mental_health_rating'] = df_social_expanded["mental_health_rating"]
    
    # Relationship data from social dataset
    # The social dataset has 'Relationship Status' according to search
    # Let's ensure we use that directly if available, otherwise use conflicts as proxy
    if 'Relationship_Status' in df_social_expanded.columns:
        df['Relationship_Status'] = df_social_expanded['Relationship_Status']
    else:
        # Fallback to simulation or just use what we have
        df['Relationship_Status'] = df_social_expanded.get('Relationship_Conflicts', 'Single')
    
    # If social dataset has Addiction Level, use it as proxy for Movie_Addiction or Social_Media_Addiction
    if 'Social_Media_Addiction_Level' in df_social_expanded.columns:
        df['Social_Media_Addiction_Level'] = df_social_expanded['Social_Media_Addiction_Level']
        # Derive Movie_Addiction from social addiction if no specific movie data
        df['Movie_Addiction'] = df['Social_Media_Addiction_Level']
    
    if 'Social_Age' in df_social_expanded.columns:
        df['age'] = df_social_expanded['Social_Age']
    
    # Try to extract Movie_Hours proxy
    if 'Primary_Platform' in df_social_expanded.columns:
        video_platforms = ['Netflix', 'YouTube', 'TikTok', 'Instagram']
        df['Movie_Hours'] = df_social_expanded.apply(
            lambda x: x['Social_Media_Hours'] * 0.4 if x['Primary_Platform'] in video_platforms else x['Social_Media_Hours'] * 0.1,
            axis=1
        )
    else:
        df['Movie_Hours'] = df['Social_Media_Hours'] * 0.3
        
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
            df[col].fillna(df[col].median(), inplace=True)
    
    # For categorical columns, fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].mode()[0], inplace=True)
            
    # Normalize column names - ensure we don't have confusing duplicates
    # Most mapping already done in load_data, just ensure case consistency
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
    """
    Split dataframe into features (X) and target (y).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def scale_features(X_train, X_test):
    """
    Standardize features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def prepare_data_for_classification(y, threshold=60):
    """
    Convert continuous scores to binary classification (Pass/Fail).
    """
    return (y >= threshold).astype(int)


def get_data_summary(df):
    """
    Generate a summary of the dataset.
    """
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
