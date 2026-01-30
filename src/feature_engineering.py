"""
Feature Engineering Module
Adds simulated columns for social media usage, movie addiction, and relationship status.
"""

import pandas as pd
import numpy as np


def add_social_media_hours(df, random_state=42):
    """
    Use real social media hours if available, otherwise simulate.
    """
    if 'Social_Media_Hours' in df.columns:
        return df['Social_Media_Hours']
        
    np.random.seed(random_state)
    n = len(df)
    base_hours = np.random.uniform(1, 8, n)
    social_media_hours = np.clip(base_hours, 0.5, 10)
    return np.round(social_media_hours, 1)


def add_movie_addiction(df, random_state=42):
    """
    Simulate movie addiction level (Low, Medium, High).
    
    Logic:
    - Correlated with social media usage patterns
    - Students with lower motivation may have higher movie addiction
    """
    np.random.seed(random_state)
    
    n = len(df)
    
    # Base probabilities
    probabilities = np.random.random(n)
    
    # If we have real Movie_Hours (from netflix_hours), use it to categorize addiction
    if 'Movie_Hours' in df.columns:
        categories = []
        for hours in df['Movie_Hours']:
            if hours < 1.5:
                categories.append('Low')
            elif hours < 3.5:
                categories.append('Medium')
            else:
                categories.append('High')
        return categories

    # Otherwise fall back to simulation based on motivation level
    if 'Motivation_Level' in df.columns:
        motivation_map = {'Low': 0.3, 'Medium': 0, 'High': -0.3}
        if df['Motivation_Level'].dtype == 'object':
            motivation_adjustment = df['Motivation_Level'].map(motivation_map).fillna(0)
        else:
            # Already encoded
            motivation_adjustment = (1 - df['Motivation_Level'] / 2) * 0.3 - 0.15
        probabilities = probabilities + motivation_adjustment
    
    # Categorize
    categories = []
    for p in probabilities:
        if p < 0.33:
            categories.append('Low')
        elif p < 0.66:
            categories.append('Medium')
        else:
            categories.append('High')
    
    return categories


def add_relationship_status(df, random_state=42):
    """
    Simulate relationship status (Single, In Relationship).
    
    Logic:
    - Random distribution with slight correlations
    - ~55% Single, ~45% In Relationship (typical for students)
    """
    np.random.seed(random_state)
    
    n = len(df)
    
    # Base probability of being in a relationship
    probabilities = np.random.random(n)
    
    # Slight adjustment based on extracurricular activities
    if 'Extracurricular_Activities' in df.columns:
        if df['Extracurricular_Activities'].dtype == 'object':
            extra_adjustment = df['Extracurricular_Activities'].map({'Yes': 0.1, 'No': -0.1}).fillna(0)
        else:
            extra_adjustment = (df['Extracurricular_Activities'] - 0.5) * 0.2
        probabilities = probabilities + extra_adjustment
    
    # Categorize (45% threshold for being in relationship)
    status = ['In Relationship' if p > 0.55 else 'Single' for p in probabilities]
    
    return status


def add_gym_discipline(df, random_state=42):
    """
    Create gym discipline score based on Physical_Activity.
    Enhances the existing Physical_Activity column with a categorical interpretation.
    """
    if 'Physical_Activity' not in df.columns:
        np.random.seed(random_state)
        return np.random.choice(['Low', 'Medium', 'High'], len(df))
    
    # Map physical activity to gym discipline
    disciplines = []
    for activity in df['Physical_Activity']:
        if activity <= 1:
            disciplines.append('Low')
        elif activity <= 3:
            disciplines.append('Medium')
        else:
            disciplines.append('High')
    
    return disciplines


def add_all_simulated_features(df, random_state=42):
    """
    Add all simulated features to the dataset.
    """
    df = df.copy()
    
    # Add simulated columns only if not present
    if 'Social_Media_Hours' not in df.columns:
        df['Social_Media_Hours'] = add_social_media_hours(df, random_state)
        
    if 'Movie_Addiction' not in df.columns:
        df['Movie_Addiction'] = add_movie_addiction(df, random_state)
        
    if 'Relationship_Status' not in df.columns:
        df['Relationship_Status'] = add_relationship_status(df, random_state)
        
    if 'Gym_Discipline' not in df.columns:
        df['Gym_Discipline'] = add_gym_discipline(df, random_state)
    
    return df


def get_focus_features():
    """
    Return the list of features we're focusing on for analysis.
    """
    return [
        'Sleep_Hours',
        'Social_Media_Hours',
        'Movie_Addiction',
        'Relationship_Status',
        'Gym_Discipline',
        'mental_health_rating',
        'Physical_Activity'
    ]
def get_all_features():
    """
    Return all features in the enhanced dataset.
    """
    return [
        # Original features
        'Hours_Studied', 'Attendance', 'Parental_Involvement',
        'Access_to_Resources', 'Extracurricular_Activities', 'Sleep_Hours',
        'Previous_Scores', 'Motivation_Level', 'Internet_Access',
        'Tutoring_Sessions', 'Family_Income', 'Teacher_Quality',
        'School_Type', 'Peer_Influence', 'Physical_Activity',
        'Learning_Disabilities', 'Parental_Education_Level',
        'Distance_from_Home', 'Gender',
        # Enhanced/Real features
        'Social_Media_Hours', 'Movie_Addiction', 
        'Relationship_Status', 'Gym_Discipline',
        'mental_health_rating', 'Social_Media_Addiction_Level'
    ]

