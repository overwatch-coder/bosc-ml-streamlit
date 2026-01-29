import pandas as pd
import numpy as np
import os
import joblib
import sys

sys.path.append(os.getcwd())

from src.data_loader import load_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.impute import SimpleImputer

# Ensuring models directory exists
os.makedirs("models", exist_ok=True)

def train_and_save_models():
    print("Loading Data...")
    df = load_data()
    
    target_col = "Exam_Score"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Categorical Features: {len(categorical_features)}")
    print(f"Numerical Features: {len(numerical_features)}")
    
    # Preprocessing Pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop' 
    )
    
    print("Splitting Data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Prepare target for classification (Pass/Fail)
    pass_threshold = 60
    y_train_clf = (y_train >= pass_threshold).astype(int)
    y_test_clf = (y_test >= pass_threshold).astype(int)
    
    print("\nTraining Regression Models...")
    
    reg_models = {
        "linear_regression": LinearRegression(),
        "random_forest_regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "mlp_regressor": MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    for name, model in reg_models.items():
        print(f"  Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"    MAE: {mae:.2f}, R2: {r2:.4f}")
        joblib.dump(pipeline, f"models/{name}.pkl")

    print("\nTraining Classification Models...")
    
    clf_models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest_classifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "knn": KNeighborsClassifier(n_neighbors=5),
        "svm": SVC(probability=True, random_state=42),
        "mlp_classifier": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
    }
    
    for name, model in clf_models.items():
        print(f"  Training {name}...")
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train_clf)
        
        y_pred = pipeline.predict(X_test)
        acc = accuracy_score(y_test_clf, y_pred)
        
        print(f"    Accuracy: {acc:.4f}")
        joblib.dump(pipeline, f"models/{name}.pkl")

    print("\nTraining Clustering Model (K-Means)...")
    kmeans_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('kmeans', KMeans(n_clusters=3, random_state=42, n_init=10))
    ])
    kmeans_pipeline.fit(X) # Use full data for clustering
    joblib.dump(kmeans_pipeline, "models/kmeans.pkl")
    print("  K-Means model saved.")
    
    # Save the preprocessor separately just in case specific use cases need it
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    
    # Save metadata about features for the app to use
    metadata = {
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "target_col": target_col,
        "feature_names": X.columns.tolist()
    }
    joblib.dump(metadata, "models/metadata.pkl")
    
    print("\nAll models trained and saved to 'models/' directory.")

if __name__ == "__main__":
    train_and_save_models()
