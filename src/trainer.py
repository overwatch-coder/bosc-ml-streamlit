import pandas as pd
import numpy as np
import os
import joblib
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
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    silhouette_score
)

def train_and_save_selected(df, selected_features, target_col='Exam_Score', models_dir='models', 
                            test_size=0.2, pass_threshold=60, n_neighbors=5):
    """
    Trains all models using selected features and saves comprehensive evaluation metrics.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Filter to selected features
    X = df[selected_features]
    y = df[target_col]
    
    # Identify feature types
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Define Preprocessing
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
    
    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    y_train_clf = (y_train >= pass_threshold).astype(int)
    y_test_clf = (y_test >= pass_threshold).astype(int)
    
    # Store evaluation results
    evaluation_results = {
        'regression': [],
        'classification': [],
        'clustering': {}
    }
    
    # Regression Models
    reg_models = {
        "linear_regression": ("Linear Regression", LinearRegression()),
        "random_forest_regressor": ("Random Forest Regressor", RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)),
        "mlp_regressor": ("MLP Regressor", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42))
    }
    
    for name, (display_name, model) in reg_models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
        pipeline.fit(X_train, y_train)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_train_pred = pipeline.predict(X_train)
        
        # Basic Metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2_test = r2_score(y_test, y_pred)
        r2_train = r2_score(y_train, y_train_pred)
        rmse = np.sqrt(mse)
        
        # Calculate Adjusted R2
        n = X_test.shape[0] # number of test samples
        p = X_test.shape[1] # number of features
        
        # Avoid division by zero
        if n > p + 1:
            adj_r2 = 1 - (1 - r2_test) * (n - 1) / (n - p - 1)
        else:
            adj_r2 = r2_test # Fallback
            
        evaluation_results['regression'].append({
            'Model Name': display_name,
            'model_key': name,
            'R² Score': r2_test,
            'Training R²': r2_train,
            'Adjusted R²': adj_r2,
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'predictions': y_pred.tolist()
        })
        
        # Save coefficients for linear regression
        if name == 'linear_regression':
            try:
                feature_names_out = pipeline.named_steps['preprocessor'].get_feature_names_out()
                coefficients = dict(zip(feature_names_out, model.coef_))
                evaluation_results['linear_coefficients'] = coefficients
            except:
                pass
        
        joblib.dump(pipeline, os.path.join(models_dir, f"{name}.pkl"))

    # Classification Models
    clf_models = {
        "logistic_regression": ("Logistic Regression", LogisticRegression(max_iter=500, random_state=42)),
        "random_forest_classifier": ("Random Forest Classifier", RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)),
        "knn": ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=n_neighbors)),
        "svm": ("Support Vector Machine", SVC(probability=True, random_state=42, cache_size=1000)),
        "mlp_classifier": ("MLP Classifier", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42))
    }
    
    for name, (display_name, model) in clf_models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
        pipeline.fit(X_train, y_train_clf)
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        y_train_pred = pipeline.predict(X_train)
        
        # Metrics
        acc_test = accuracy_score(y_test_clf, y_pred)
        acc_train = accuracy_score(y_train_clf, y_train_pred)
        prec = precision_score(y_test_clf, y_pred, zero_division=0)
        rec = recall_score(y_test_clf, y_pred, zero_division=0)
        f1 = f1_score(y_test_clf, y_pred, zero_division=0)
        cm = confusion_matrix(y_test_clf, y_pred)
        
        evaluation_results['classification'].append({
            'Model Name': display_name,
            'model_key': name,
            'Accuracy': acc_test,
            'Training Accuracy': acc_train,
            'Precision': prec,
            'Recall': rec,
            'F1 Score': f1,
            'confusion_matrix': cm.tolist()
        })
        
        joblib.dump(pipeline, os.path.join(models_dir, f"{name}.pkl"))

    # 3. Clustering
    kmeans_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('kmeans', KMeans(n_clusters=3, random_state=42, n_init=5))
    ])
    kmeans_pipeline.fit(X)
    
    # Get cluster labels and centers
    X_transformed = kmeans_pipeline.named_steps['preprocessor'].transform(X)
    cluster_labels = kmeans_pipeline.named_steps['kmeans'].labels_
    cluster_centers = kmeans_pipeline.named_steps['kmeans'].cluster_centers_
    
    # Calculate silhouette score
    sil_score = silhouette_score(X_transformed, cluster_labels)
    
    # Cluster sizes
    unique, counts = np.unique(cluster_labels, return_counts=True)
    cluster_sizes = dict(zip(unique.tolist(), counts.tolist()))
    
    evaluation_results['clustering'] = {
        'n_clusters': 3,
        'silhouette_score': sil_score,
        'inertia': kmeans_pipeline.named_steps['kmeans'].inertia_,
        'cluster_sizes': cluster_sizes,
        'cluster_centers': cluster_centers.tolist(),
        'cluster_labels': cluster_labels.tolist()
    }
    
    joblib.dump(kmeans_pipeline, os.path.join(models_dir, "kmeans.pkl"))
    
    # Save evaluation results
    joblib.dump(evaluation_results, os.path.join(models_dir, "evaluation_results.pkl"))
    
    # Save test data for visualizations
    test_data = {
        'y_test': y_test.tolist(),
        'y_test_clf': y_test_clf.tolist()
    }
    joblib.dump(test_data, os.path.join(models_dir, "test_data.pkl"))
    
    # Metadata
    metadata = {
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "target_col": target_col,
        "feature_names": selected_features,
        "pass_threshold": pass_threshold,
        "test_size": test_size,
        "n_neighbors": n_neighbors
    }
    joblib.dump(metadata, os.path.join(models_dir, "metadata.pkl"))
    
    return True
