"""
Visualizations Module
Creates charts and plots for data exploration and model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def create_correlation_heatmap(df, figsize=(12, 10)):
    """
    Create a correlation heatmap for numerical features.
    """
    # Select only numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numerical_df.corr()
    
    # Create heatmap using plotly
    fig = px.imshow(
        corr_matrix,
        text_auto='.2f',
        aspect='auto',
        color_continuous_scale='RdBu_r',
        title='Feature Correlation Heatmap'
    )
    
    fig.update_layout(
        width=800,
        height=700,
        title_font_size=16
    )
    
    return fig


def create_feature_importance_chart(coefficients, title="Feature Importance (Linear Regression Coefficients)"):
    """
    Create a bar chart showing feature importance/coefficients.
    """
    if not coefficients:
        return None
    
    # Sort by absolute value
    sorted_coefs = dict(sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True))
    
    fig = go.Figure(go.Bar(
        x=list(sorted_coefs.values()),
        y=list(sorted_coefs.keys()),
        orientation='h',
        marker_color=['green' if v > 0 else 'red' for v in sorted_coefs.values()]
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Coefficient Value',
        yaxis_title='Feature',
        height=500,
        yaxis={'categoryorder': 'total ascending'}
    )
    
    return fig


def create_distribution_plot(df, column, title=None):
    """
    Create a distribution/histogram plot for a column.
    """
    fig = px.histogram(
        df, 
        x=column, 
        marginal='box',
        title=title or f'Distribution of {column}',
        color_discrete_sequence=['#636EFA']
    )
    
    fig.update_layout(
        showlegend=False,
        height=400
    )
    
    return fig


def create_scatter_with_regression(df, x_col, y_col, color_col=None):
    """
    Create a scatter plot with regression line.
    """
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        trendline='ols',
        title=f'{x_col} vs {y_col}',
        opacity=0.6
    )
    
    fig.update_layout(height=500)
    
    return fig


def create_box_plot(df, x_col, y_col, title=None):
    """
    Create a box plot comparing categories.
    """
    fig = px.box(
        df,
        x=x_col,
        y=y_col,
        title=title or f'{y_col} by {x_col}',
        color=x_col
    )
    
    fig.update_layout(height=450)
    
    return fig


def create_model_comparison_chart(comparison_df, metric='Accuracy'):
    """
    Create a bar chart comparing model performances.
    """
    if comparison_df.empty:
        return None
    
    fig = px.bar(
        comparison_df,
        x='Model',
        y=metric,
        title=f'Model Comparison - {metric}',
        color='Model',
        text=metric
    )
    
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        showlegend=False,
        height=450,
        yaxis_range=[0, 1.1] if metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'R² Score'] else None
    )
    
    return fig


def create_confusion_matrix_plot(cm, labels=['Fail', 'Pass']):
    """
    Create a confusion matrix heatmap.
    """
    fig = px.imshow(
        cm,
        text_auto=True,
        x=labels,
        y=labels,
        color_continuous_scale='Blues',
        title='Confusion Matrix'
    )
    
    fig.update_layout(
        xaxis_title='Predicted',
        yaxis_title='Actual',
        height=400,
        width=400
    )
    
    return fig


def create_cluster_visualization(df, cluster_labels, x_col, y_col):
    """
    Create a scatter plot showing cluster assignments.
    """
    df_plot = df.copy()
    df_plot['Cluster'] = cluster_labels.astype(str)
    
    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        color='Cluster',
        title=f'K-Means Clustering: {x_col} vs {y_col}',
        opacity=0.7
    )
    
    fig.update_layout(height=500)
    
    return fig


def create_loss_curve(loss_values, title="Training Loss Curve"):
    """
    Create a line plot showing the neural network training loss.
    """
    if loss_values is None:
        return None
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=loss_values,
        mode='lines',
        name='Loss',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Iteration',
        yaxis_title='Loss',
        height=400
    )
    
    return fig


def create_actual_vs_predicted(y_actual, y_predicted, title="Actual vs Predicted Scores"):
    """
    Create a scatter plot comparing actual vs predicted values.
    """
    fig = go.Figure()
    
    # Scatter plot of predictions
    fig.add_trace(go.Scatter(
        x=y_actual,
        y=y_predicted,
        mode='markers',
        name='Predictions',
        marker=dict(color='blue', opacity=0.5)
    ))
    
    # Perfect prediction line
    min_val = min(min(y_actual), min(y_predicted))
    max_val = max(max(y_actual), max(y_predicted))
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        name='Perfect Prediction',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Actual Score',
        yaxis_title='Predicted Score',
        height=500
    )
    
    return fig


def create_feature_vs_target_grid(df, features, target='Exam_Score'):
    """
    Create a grid of scatter plots showing each feature vs target.
    """
    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=features
    )
    
    for i, feature in enumerate(features):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        fig.add_trace(
            go.Scatter(
                x=df[feature],
                y=df[target],
                mode='markers',
                marker=dict(opacity=0.5),
                name=feature
            ),
            row=row,
            col=col
        )
    
    fig.update_layout(
        height=300 * n_rows,
        showlegend=False,
        title_text=f"Features vs {target}"
    )
    
    return fig


def create_pie_chart(series, title="Distribution"):
    """
    Create a pie chart for categorical data.
    """
    fig = px.pie(
        values=series.value_counts().values,
        names=series.value_counts().index,
        title=title
    )
    
    fig.update_layout(height=400)
    
    return fig


def create_metrics_radar_chart(comparison_df):
    """
    Create a radar chart comparing models across multiple metrics.
    """
    if comparison_df.empty:
        return None
    
    metrics = [col for col in comparison_df.columns if col != 'Model']
    
    fig = go.Figure()
    
    for _, row in comparison_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[m] for m in metrics],
            theta=metrics,
            fill='toself',
            name=row['Model']
        ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title='Model Comparison Radar Chart',
        height=500
    )
    
    return fig
