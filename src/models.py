"""
Machine Learning Models Module
Implements all ML models for student performance prediction.

Models:
- Linear Regression (Regression)
- Logistic Regression (Classification)
- K-Nearest Neighbors (Classification)
- Support Vector Machine (Classification)
- K-Means Clustering
- Neural Network - MLP (Classification/Regression)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)
from sklearn.preprocessing import StandardScaler


class StudentPerformanceModels:
    """
    A class containing all ML models for student performance prediction.
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def train_linear_regression(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate Linear Regression model.
        Used for predicting exact exam scores.
        """
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        results = {
            'model_name': 'Linear Regression',
            'model_type': 'Regression',
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'coefficients': dict(zip(X_train.columns if hasattr(X_train, 'columns') else range(X_train.shape[1]), model.coef_)),
            'intercept': model.intercept_,
            'predictions': y_pred_test
        }
        
        self.models['linear_regression'] = model
        self.results['linear_regression'] = results
        return results
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate Logistic Regression model.
        Used for Pass/Fail classification.
        """
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        results = {
            'model_name': 'Logistic Regression',
            'model_type': 'Classification',
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'predictions': y_pred_test
        }
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = results
        return results
    
    def train_knn(self, X_train, y_train, X_test, y_test, n_neighbors=5):
        """
        Train and evaluate K-Nearest Neighbors classifier.
        """
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        results = {
            'model_name': f'KNN (k={n_neighbors})',
            'model_type': 'Classification',
            'n_neighbors': n_neighbors,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'predictions': y_pred_test
        }
        
        self.models['knn'] = model
        self.results['knn'] = results
        return results
    
    def train_svm(self, X_train, y_train, X_test, y_test, kernel='rbf'):
        """
        Train and evaluate Support Vector Machine classifier.
        """
        model = SVC(kernel=kernel, random_state=42)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        results = {
            'model_name': f'SVM ({kernel} kernel)',
            'model_type': 'Classification',
            'kernel': kernel,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'predictions': y_pred_test
        }
        
        self.models['svm'] = model
        self.results['svm'] = results
        return results
    
    def train_kmeans(self, X, n_clusters=3):
        """
        Train K-Means clustering to group students.
        """
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = model.fit_predict(X)
        
        # Calculate silhouette score
        sil_score = silhouette_score(X, clusters) if len(set(clusters)) > 1 else 0
        
        # Metrics
        results = {
            'model_name': f'K-Means (k={n_clusters})',
            'model_type': 'Clustering',
            'n_clusters': n_clusters,
            'cluster_labels': clusters,
            'cluster_centers': model.cluster_centers_,
            'inertia': model.inertia_,
            'silhouette_score': sil_score,
            'cluster_sizes': pd.Series(clusters).value_counts().to_dict()
        }
        
        self.models['kmeans'] = model
        self.results['kmeans'] = results
        return results
    
    def train_mlp_classifier(self, X_train, y_train, X_test, y_test, 
                             hidden_layers=(100, 50), max_iter=500):
        """
        Train and evaluate Neural Network (MLP) classifier.
        """
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        results = {
            'model_name': f'Neural Network MLP {hidden_layers}',
            'model_type': 'Classification (Deep Learning)',
            'hidden_layers': hidden_layers,
            'n_iterations': model.n_iter_,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'test_accuracy': accuracy_score(y_test, y_pred_test),
            'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred_test),
            'predictions': y_pred_test,
            'loss_curve': model.loss_curve_ if hasattr(model, 'loss_curve_') else None
        }
        
        self.models['mlp_classifier'] = model
        self.results['mlp_classifier'] = results
        return results
    
    def train_mlp_regressor(self, X_train, y_train, X_test, y_test,
                            hidden_layers=(100, 50), max_iter=500):
        """
        Train and evaluate Neural Network (MLP) regressor for score prediction.
        """
        model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Metrics
        results = {
            'model_name': f'Neural Network MLP Regressor {hidden_layers}',
            'model_type': 'Regression (Deep Learning)',
            'hidden_layers': hidden_layers,
            'n_iterations': model.n_iter_,
            'train_r2': r2_score(y_train, y_pred_train),
            'test_r2': r2_score(y_test, y_pred_test),
            'train_mse': mean_squared_error(y_train, y_pred_train),
            'test_mse': mean_squared_error(y_test, y_pred_test),
            'train_mae': mean_absolute_error(y_train, y_pred_train),
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'predictions': y_pred_test,
            'loss_curve': model.loss_curve_ if hasattr(model, 'loss_curve_') else None
        }
        
        self.models['mlp_regressor'] = model
        self.results['mlp_regressor'] = results
        return results
    
    def get_model_comparison(self, model_type='classification'):
        """
        Get a comparison of all trained models of a specific type.
        """
        comparison = []
        
        for name, result in self.results.items():
            if model_type == 'classification' and 'accuracy' in str(result.get('model_type', '')).lower():
                continue
            if model_type == 'classification' and 'test_accuracy' in result:
                comparison.append({
                    'Model': result['model_name'],
                    'Accuracy': result['test_accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1 Score': result['f1_score']
                })
            elif model_type == 'regression' and 'test_r2' in result:
                comparison.append({
                    'Model': result['model_name'],
                    'R² Score': result['test_r2'],
                    'MSE': result['test_mse'],
                    'MAE': result['test_mae']
                })
        
        return pd.DataFrame(comparison)
    
    def predict(self, model_name, X):
        """
        Make predictions using a trained model.
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found. Train it first.")
        
        return self.models[model_name].predict(X)
    
    def get_feature_importance(self, model_name='linear_regression'):
        """
        Get feature importance from the model.
        """
        if model_name == 'linear_regression' and model_name in self.results:
            return self.results[model_name].get('coefficients', {})
        return {}


def train_all_models(X_train, y_train_reg, X_test, y_test_reg, 
                     y_train_clf, y_test_clf, X_full=None):
    """
    Train all models and return results.
    
    Parameters:
    - X_train, X_test: Feature sets
    - y_train_reg, y_test_reg: Continuous target (exam scores)
    - y_train_clf, y_test_clf: Binary target (pass/fail)
    - X_full: Full feature set for clustering
    """
    models = StudentPerformanceModels()
    
    # Regression models
    models.train_linear_regression(X_train, y_train_reg, X_test, y_test_reg)
    models.train_mlp_regressor(X_train, y_train_reg, X_test, y_test_reg)
    
    # Classification models
    models.train_logistic_regression(X_train, y_train_clf, X_test, y_test_clf)
    models.train_knn(X_train, y_train_clf, X_test, y_test_clf)
    models.train_svm(X_train, y_train_clf, X_test, y_test_clf)
    models.train_mlp_classifier(X_train, y_train_clf, X_test, y_test_clf)
    
    # Clustering
    if X_full is not None:
        models.train_kmeans(X_full)
    
    return models
