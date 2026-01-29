"""
Student Performance Prediction Dashboard
A comprehensive ML solution with interactive visualizations.

"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
import joblib

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import (
    load_data, clean_data, encode_categorical
)
from src.trainer import train_and_save_selected
from src.visualizations import (
    create_correlation_heatmap, create_feature_importance_chart,
    create_distribution_plot, create_scatter_with_regression,
    create_box_plot, create_model_comparison_chart,
    create_confusion_matrix_plot, create_cluster_visualization,
    create_loss_curve, create_actual_vs_predicted, create_pie_chart,
    create_metrics_radar_chart
)
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_enhanced' not in st.session_state:
    st.session_state.df_enhanced = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'model_handler' not in st.session_state:
    st.session_state.model_handler = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = [
        'Attendance', 'Hours_Studied', 'Previous_Scores', 'Tutoring_Sessions', 
        'Physical_Activity', 'Movie_Hours', 'Sleep_Hours', 'Movie_Addiction', 
        'mental_health_rating', 'Social_Media_Hours', 'Relationship_Status', 
        'Extracurricular_Activities', 'Gym_Discipline', 'Motivation_Level'
    ]


def load_and_prepare_data():
    """Load and prepare the dataset."""
    with st.spinner("📥 Loading dataset..."):
        try:
            df = load_data()
            df = clean_data(df)
            st.session_state.df = df
            st.session_state.df_enhanced = df
            st.session_state.data_loaded = True
            return True
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False


@st.dialog("🎯 Configure Factors of Interest")
def feature_selection_dialog():
    """Modal to select active features for analysis."""
    st.write("Which student factors would you like to analyze? Factors not selected will be **dropped** from visualizations and model training.")
    
    df_full = st.session_state.df_enhanced
    all_cols = sorted([c for c in df_full.columns if c != 'Exam_Score'])
    
    # Default features we want to use (if they exist)
    default_features = [
        'Attendance', 'Hours_Studied', 'Previous_Scores', 'Tutoring_Sessions', 
        'Physical_Activity', 'Movie_Hours', 'Sleep_Hours', 'Movie_Addiction', 
        'mental_health_rating', 'Social_Media_Hours', 'Relationship_Status', 
        'Extracurricular_Activities', 'Gym_Discipline', 'Motivation_Level'
    ]
    
    # Filter to only include features that actually exist in the dataset
    valid_defaults = [f for f in default_features if f in all_cols]
    
    # Auto-initialize if empty (doubling up for safety)
    if not st.session_state.selected_features:
        st.session_state.selected_features = valid_defaults if valid_defaults else all_cols[:10]

    # Ensure current selection only contains valid features
    current_selection = [f for f in st.session_state.selected_features if f in all_cols]
    if len(current_selection) != len(st.session_state.selected_features):
        st.session_state.selected_features = current_selection

    # Selection
    new_selection = st.multiselect(
        "Active Factors (Factors of Interest)",
        options=all_cols,
        default=st.session_state.selected_features
    )
    
    # Show dropped summary
    dropped = list(set(all_cols) - set(new_selection))
    if dropped:
        with st.expander(f"🗑️ View {len(dropped)} Dropped Factors"):
            st.write(", ".join(dropped))
            
    if st.button("✅ Apply Selection & Close", width='stretch', type="primary"):
        st.session_state.selected_features = new_selection
        # Reset models since feature set changed
        st.session_state.models_trained = False
        st.session_state.model_handler = None
        st.rerun()


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/graduation-cap.png", width=80)
        st.title("🎓 Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home", "📊 Data Exploration", "📈 Visualizations", 
             "🤖 Model Training", "🎯 Predictions"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 👥 Group Members")
        st.markdown("""
        - Atsu M. Nyamadi
        - Celestin HAKORIMANA
        - Desange UWERA
        - Mariam Oukadour
        - Rim Abdelhakmi
        """)
        
        st.markdown("---")
        st.markdown("### 📁 Dataset")
        st.markdown("- [Student Performance Factors](https://www.kaggle.com/datasets/ayeshaseherr/student-performance)")

        if st.session_state.data_loaded:
            st.markdown("---")
            # Show a mini-summary of dropped features for awareness
            df_full = st.session_state.df_enhanced
            all_cols = [c for c in df_full.columns if c != 'Exam_Score']
            dropped = list(set(all_cols) - set(st.session_state.selected_features))
            if dropped:
                st.caption(f"⚠️ {len(dropped)} factors are currently dropped.")
    
    # Main content based on page selection
    if page == "🏠 Home":
        render_home_page()
    elif page == "📊 Data Exploration":
        render_data_exploration_page()
    elif page == "📈 Visualizations":
        render_visualizations_page()
    elif page == "🤖 Model Training":
        render_model_training_page()
    elif page == "🎯 Predictions":
        render_predictions_page()
    elif page == "📋 Reports":
        render_report_page()


def render_home_page():
    """Render the home page."""
    st.markdown('<p class="main-header">🎓 Student Performance Prediction</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">A Machine Learning approach to predict academic performance</p>', unsafe_allow_html=True)
    
    # Layout for data loading button
    center_col1, center_col2, center_col3 = st.columns([1, 2, 1])
    with center_col2:
        if not st.session_state.data_loaded:
            if st.button("📥 Load Dataset from Kaggle", width='stretch', type="primary"):
                if load_and_prepare_data():
                    # After success, trigger the dialog immediately for first-time selection
                    st.rerun()
        else:
            st.success("✅ Dataset loaded successfully!")
            if st.button("🎯 Toggle & Select Factors", width='stretch', type="primary"):
                feature_selection_dialog()
            st.caption("Click the button above anytime to change the focus of your analysis.")
    
    st.markdown("---")
    
    # Project Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📌 Project Overview")
        st.markdown("""
        This project analyzes factors affecting student academic performance and builds 
        predictive models using **real data** from two integrated Kaggle datasets.
        
        **Key Factors of Interest (Active):**
        """)
        selected_features = st.session_state.get('selected_features', [])
        if selected_features:
            for feat in selected_features[:10]: # Show top 10
                st.markdown(f"- ✨ **{feat.replace('_', ' ')}**")
            if len(selected_features) > 10:
                st.caption(f"... and {len(selected_features)-10} more")
        else:
            st.caption("No features selected in sidebar.")
    
    with col2:
        st.markdown("### 🤖 ML Models Implemented")
        st.markdown("""
        | Model | Type |
        |-------|------|
        | Linear Regression | Regression |
        | Random Forest | Regression/Classification |
        | Logistic Regression | Classification |
        | K-Nearest Neighbors | Classification |
        | Support Vector Machine | Classification |
        | K-Means | Clustering |
        | Neural Network (MLP) | Deep Learning |
        """)
    
    # Dataset summary if loaded
    if st.session_state.data_loaded:
        st.markdown("---")
        st.markdown("### 📊 Dataset Summary")
        
        df = st.session_state.df_enhanced
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Students", f"{len(df):,}")
        with col2:
            count = len(st.session_state.selected_features)
            st.metric("Active Features", count)
        with col3:
            st.metric("Avg Exam Score", f"{df['Exam_Score'].mean():.1f}")
        with col4:
            st.metric("Pass Rate (≥60)", f"{(df['Exam_Score'] >= 60).mean()*100:.1f}%")


def render_data_exploration_page():
    """Render the data exploration page."""
    st.markdown("## 📊 Data Exploration")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first from the Home page.")
        return
    
    df_full = st.session_state.df_enhanced
    
    # Validate selected features exist in dataset
    available_cols = df_full.columns.tolist()
    selected_features = st.session_state.selected_features
    
    # Filter out features that don't exist in the dataset
    valid_features = [f for f in selected_features if f in available_cols]
    
    # Update session state if features were filtered
    if len(valid_features) != len(selected_features):
        invalid_features = [f for f in selected_features if f not in available_cols]
        st.warning(f"⚠️ Some selected features are not in the dataset and were removed: {', '.join(invalid_features)}")
        st.session_state.selected_features = valid_features
        selected_features = valid_features
    
    # Always include target if it exists
    if 'Exam_Score' in available_cols:
        cols_to_show = selected_features + ['Exam_Score']
    else:
        cols_to_show = selected_features
        st.error("Target column 'Exam_Score' not found in dataset!")
    
    df = df_full[cols_to_show]
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "📈 Statistics", "❓ Missing Values", "🔧 Extra Features"])
    
    with tab1:
        st.markdown("### Raw Data Preview")
        st.dataframe(df.head(100), width='stretch')
        st.info(f"Showing first 100 rows of {len(df):,} total records")
    
    with tab2:
        st.markdown("### Descriptive Statistics")
        st.dataframe(df.describe().round(2), width='stretch')
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Numerical Columns")
            st.write(list(df.select_dtypes(include=[np.number]).columns))
        with col2:
            st.markdown("#### Categorical Columns")
            st.write(list(df.select_dtypes(include=['object']).columns))
    
    with tab3:
        st.markdown("### Missing Values Analysis")
        missing = df.isnull().sum()
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            'Missing %': (missing.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df[missing_df['Missing Count'] > 0], width='stretch')
        
        if missing.sum() == 0:
            st.success("✅ No missing values in the dataset!")
    
    with tab4:
        st.markdown("### Extra Features")
        st.info("""
        The following features were added to align with the project's focus on 
        counter-intuitive factors affecting academic performance:
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **📱 Social Media Hours**
            - Range: 0.5 - 10 hours/day
            - Inversely correlated with sleep and study hours
            """)
            
            st.markdown("""
            **🎬 Movie Addiction**
            - Categories: Low, Medium, High
            - Influenced by motivation level
            """)
        
        with col2:
            st.markdown("""
            **💑 Relationship Status**
            - Categories: Single, In Relationship
            - ~55% Single, ~45% In Relationship
            """)
            
            st.markdown("""
            **🏋️ Gym Discipline**
            - Categories: Low, Medium, High
            - Based on Physical Activity levels
            """)
        
        # Show distribution of extra features
        st.markdown("#### Distribution of Extra Features")
        
        extra_cols = ['Social_Media_Hours', 'Movie_Addiction', 'Relationship_Status', 'Gym_Discipline']
        for i, col in enumerate(extra_cols):
            if col in df.columns:
                if df[col].dtype == 'object':
                    st.plotly_chart(create_pie_chart(df[col], col), width='stretch', key=f"extra_pie_{col}")
                else:
                    st.plotly_chart(create_distribution_plot(df, col), width='stretch', key=f"extra_dist_{col}")
                st.markdown("---") # Add a separator between vertical plots


def render_visualizations_page():
    """Render the visualizations page."""
    st.markdown("## 📈 Data Visualizations")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first from the Home page.")
        return
    
    df_full = st.session_state.df_enhanced
    selected_features = st.session_state.selected_features
    df = df_full[selected_features + ['Exam_Score']]
    
    # Encode for correlation
    df_encoded, _ = encode_categorical(df)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Correlation", "📊 Distributions", "📉 Relationships", "🎯 Focus Factors"])
    
    with tab1:
        st.markdown("### Correlation Heatmap")
        fig = create_correlation_heatmap(df_encoded)
        st.plotly_chart(fig, width='stretch', key="corr_heatmap")
        
        # Top correlations with Exam_Score
        st.markdown("### Top Correlations with Exam Score")
        corr_with_target = df_encoded.corr()['Exam_Score'].drop('Exam_Score').sort_values(ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📈 Positive Correlations**")
            st.dataframe(corr_with_target.head(5).round(3))
        with col2:
            st.markdown("**📉 Negative Correlations**")
            st.dataframe(corr_with_target.tail(5).round(3))
    
    with tab2:
        st.markdown("### Feature Distributions")
        
        col1, col2 = st.columns(2)
        with col1:
            feature = st.selectbox("Select Feature", df.columns.tolist())
        with col2:
            pass
        
        if df[feature].dtype in ['int64', 'float64']:
            fig = create_distribution_plot(df, feature)
            st.plotly_chart(fig, width='stretch', key=f"dist_{feature}")
        else:
            fig = create_pie_chart(df[feature], f"Distribution of {feature}")
            st.plotly_chart(fig, width='stretch', key=f"pie_{feature}")
    
    with tab3:
        st.markdown("### Feature vs Exam Score")
        
        col1, col2 = st.columns(2)
        with col1:
            # Allow all selected features (categorical will be plotted as encoded)
            x_feature = st.selectbox("X-axis Feature", 
                                     selected_features,
                                     key="scatter_x")
        with col2:
            # Only categorical selected features for coloring
            cat_selected = [c for c in selected_features if df_full[c].dtype == 'object']
            color_by = st.selectbox("Color by (optional)", 
                                    ['None'] + cat_selected,
                                    key="scatter_color")
        
        color_col = None if color_by == 'None' else color_by
        
        # Using encoded df for plotting
        df_plot = df_encoded.copy()
        if color_col and color_col in df.columns:
            df_plot[color_col] = df[color_col]
        
        # Create mapping for X-axis labels if feature is categorical
        x_tick_map = None
        if df[x_feature].dtype == 'object':
             unique_vals = df[[x_feature]].copy()
             unique_vals['encoded'] = df_encoded[x_feature]
             unique_vals = unique_vals.drop_duplicates().sort_values('encoded')
             x_tick_map = dict(zip(unique_vals['encoded'], unique_vals[x_feature]))

        fig = create_scatter_with_regression(df_plot, x_feature, 'Exam_Score', color_col, x_tick_map)
        st.plotly_chart(fig, width='stretch', key=f"scatter_{x_feature}")
    
    with tab4:
        st.markdown("### Focus Factors Analysis")
        st.info("Analyzing the key factors of interest: Sleep, Social Media, Movie Addiction, Relationship Status, Gym Discipline")
        
        # Box plots for categorical focus factors
        categorical_all = ['Movie_Addiction', 'Relationship_Status', 'Gym_Discipline', 'mental_health_rating', 'stress_level', 'Motivation_Level', 'Gender', 'School_Type', 'Major']
        categorical_focus = [f for f in categorical_all if f in selected_features]
        
        for i, factor in enumerate(categorical_focus):
            if factor in df.columns:
                fig = create_box_plot(df, factor, 'Exam_Score')
                st.plotly_chart(fig, width='stretch', key=f"box_{factor}")
        
        # Scatter for numerical focus factors
        st.markdown("### Numerical Factors")
        col1, col2 = st.columns(2)
        with col1:
            if 'Sleep_Hours' in df.columns:
                fig = create_scatter_with_regression(df, 'Sleep_Hours', 'Exam_Score')
                st.plotly_chart(fig, width='stretch', key="scatter_sleep")
        with col2:
            if 'Social_Media_Hours' in df.columns:
                fig = create_scatter_with_regression(df, 'Social_Media_Hours', 'Exam_Score')
                st.plotly_chart(fig, width='stretch', key="scatter_social")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'Movie_Hours' in df.columns:
                fig = create_scatter_with_regression(df, 'Movie_Hours', 'Exam_Score')
                st.plotly_chart(fig, width='stretch', key="scatter_movie")
        with col2:
            if 'exam_anxiety_score' in df.columns:
                fig = create_scatter_with_regression(df, 'exam_anxiety_score', 'Exam_Score')
                st.plotly_chart(fig, width='stretch', key="scatter_anxiety")


def render_model_training_page():
    """Render the model training page."""
    st.markdown("## 🤖 Model Status")
    
    if os.path.exists("models/metadata.pkl"):
        st.success("✅ Models have been successfully trained and saved!")
        
        st.markdown("### 📁 Available Models")
        st.markdown("""
        The following models are optimized and ready for prediction:
        
        **Regression (Score Prediction)**
        - Linear Regression
        - Random Forest Regressor
        - Neural Network (MLP) Regressor

        **Classification (Pass/Fail)**
        - Logistic Regression
        - Random Forest Classifier
        - K-Nearest Neighbors
        - Support Vector Machine
        - Neural Network (MLP) Classifier
        
        **Clustering**
        - K-Means
        """)
        
    st.markdown("### 🛠️ Retrain Models")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="train_test_size")
    with col2:
        pass_threshold = st.slider("Pass Threshold (%)", 50, 100, 60, key="train_pass_threshold")
    with col3:
        n_neighbors = st.slider("K for KNN", 3, 15, 5, key="train_knn_k")

    st.info(f"Currently focused on **{len(st.session_state.selected_features)} factors**. Retraining will optimize models specifically for these factors.")
    
    if st.session_state.df is not None:
        if st.button("🚀 Retrain Models with Active Factors", type="primary", width='stretch'):
            with st.spinner("Retraining all models... This may take a moment."):
                try:
                    success = train_and_save_selected(
                        st.session_state.df, 
                        st.session_state.selected_features,
                        test_size=test_size,
                        pass_threshold=pass_threshold,
                        n_neighbors=n_neighbors
                    )
                    if success:
                        st.success("✅ Models retrained successfully and saved to disk!")
                        st.session_state.models_trained = True
                        st.rerun()
                except Exception as e:
                    st.error(f"Error during retraining: {e}")
    else:
        st.warning("Please load the dataset on the Home page first to enable retraining.")
    
    st.markdown("---")
    
    # Enable results view if available
    if os.path.exists("models/metadata.pkl") and os.path.exists("models/evaluation_results.pkl"):
        st.session_state.models_trained = True
        
        st.markdown("---")
        st.markdown("### 📊 Model Evaluation Results")
        
        # Load evaluation results
        eval_results = joblib.load("models/evaluation_results.pkl")
        test_data = joblib.load("models/test_data.pkl") if os.path.exists("models/test_data.pkl") else None
        
        # Tabs for different model types
        tab1, tab2, tab3 = st.tabs(["📉 Regression Models", "🏷️ Classification Models", "🔮 Clustering"])
        
        with tab1:
            st.markdown("#### Regression Model Comparison")
            st.info("Comparing regression models across key metrics. Best values are highlighted in green.")
            
            if eval_results.get('regression'):
                # Create DataFrame
                reg_df = pd.DataFrame(eval_results['regression'])
                reg_display = reg_df[['Model Name', 'R² Score', 'MAE', 'MSE', 'RMSE']].copy()
                
                # Transpose for better comparison
                reg_display_t = reg_display.set_index('Model Name').T
                
                def highlight_best_reg(row):
                    is_min_metrics = ['MAE', 'MSE', 'RMSE']
                    if row.name in is_min_metrics:
                        best_val = row.min()
                    else:
                        best_val = row.max()
                    return ['background-color: rgba(0, 255, 0, 0.3)' if v == best_val else '' for v in row]
                
                st.dataframe(reg_display_t.style.apply(highlight_best_reg, axis=1).format(precision=4), use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                with col1:
                    fig = create_model_comparison_chart(reg_display, 'R² Score')
                    st.plotly_chart(fig, use_container_width=True, key="reg_r2_comp")
                with col2:
                    fig = create_model_comparison_chart(reg_display, 'MAE')
                    st.plotly_chart(fig, use_container_width=True, key="reg_mae_comp")
                
                # Actual vs Predicted for best model
                best_model = reg_display.loc[reg_display['R² Score'].idxmax()]
                st.markdown(f"#### {best_model['Model Name']}: Actual vs Predicted")
                
                if test_data:
                    best_idx = reg_display['R² Score'].idxmax()
                    predictions = eval_results['regression'][best_idx]['predictions']
                    fig = create_actual_vs_predicted(test_data['y_test'], predictions)
                    st.plotly_chart(fig, use_container_width=True, key="best_reg_actual_pred")
                
                # Feature importance for Linear Regression
                if 'linear_coefficients' in eval_results:
                    st.markdown("#### Feature Importance (Linear Regression Coefficients)")
                    coefficients = eval_results['linear_coefficients']
                    fig = create_feature_importance_chart(coefficients)
                    st.plotly_chart(fig, use_container_width=True, key="lr_feat_imp")
        
        with tab2:
            st.markdown("#### Classification Model Comparison")
            st.info("Comparing classification models. Best values are highlighted in green.")
            
            if eval_results.get('classification'):
                # Create DataFrame
                clf_df = pd.DataFrame(eval_results['classification'])
                clf_display = clf_df[['Model Name', 'Accuracy', 'Precision', 'Recall', 'F1 Score']].copy()
                
                # Transpose
                clf_display_t = clf_display.set_index('Model Name').T
                
                def highlight_best_clf(row):
                    best_val = row.max()
                    return ['background-color: rgba(0, 255, 0, 0.3)' if v == best_val else '' for v in row]
                
                st.dataframe(clf_display_t.style.apply(highlight_best_clf, axis=1).format(precision=4), use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                with col1:
                    fig = create_model_comparison_chart(clf_display, 'Accuracy')
                    st.plotly_chart(fig, use_container_width=True, key="clf_acc_comp")
                with col2:
                    fig = create_model_comparison_chart(clf_display, 'F1 Score')
                    st.plotly_chart(fig, use_container_width=True, key="clf_f1_comp")
                
                # Radar chart
                st.markdown("#### Model Comparison Radar Chart")
                fig = create_metrics_radar_chart(clf_display)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="clf_radar")
                
                # Confusion matrices
                st.markdown("#### Confusion Matrices")
                num_cols = 2
                clf_models_with_cm = [m for m in eval_results['classification'] if 'confusion_matrix' in m]
                
                for row_idx in range((len(clf_models_with_cm) + num_cols - 1) // num_cols):
                    cols = st.columns(num_cols)
                    for col_idx in range(num_cols):
                        model_idx = row_idx * num_cols + col_idx
                        if model_idx < len(clf_models_with_cm):
                            model_data = clf_models_with_cm[model_idx]
                            with cols[col_idx]:
                                st.markdown(f"**{model_data['Model Name']}**")
                                cm = np.array(model_data['confusion_matrix'])
                                fig = create_confusion_matrix_plot(cm)
                                st.plotly_chart(fig, use_container_width=True, key=f"cm_{model_data['model_key']}")
        
        with tab3:
            st.markdown("#### K-Means Clustering Results")
            
            if eval_results.get('clustering'):
                cluster_data = eval_results['clustering']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Clusters", cluster_data['n_clusters'])
                with col2:
                    st.metric("Silhouette Score", f"{cluster_data['silhouette_score']:.3f}")
                with col3:
                    st.metric("Inertia", f"{cluster_data['inertia']:.1f}")
                
                st.markdown("#### Cluster Distribution")
                cluster_sizes = cluster_data['cluster_sizes']
                
                # Create a bar chart for cluster sizes
                size_df = pd.DataFrame({
                    'Cluster': [f"Group {k}" for k in cluster_sizes.keys()],
                    'Size': list(cluster_sizes.values())
                })
                
                fig = px.bar(size_df, x='Cluster', y='Size', title='Students per Cluster',
                            color='Cluster', text='Size')
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True, key="cluster_sizes")
                
                st.markdown("#### Cluster Profiles")
                st.info("Each cluster represents a distinct student profile based on their characteristics.")
                
                # Create interpretable cluster profiles
                centers = np.array(cluster_data['cluster_centers'])
                metadata = joblib.load("models/metadata.pkl")
                
                profiles = []
                for i in range(len(centers)):
                    profile = {
                        'Cluster': f"Group {i}",
                        'Size': cluster_sizes.get(i, 0)
                    }
                    
                    # Interpret based on standardized values
                    avg_vals = centers[i]
                    
                    # Categorize cluster
                    if np.mean(avg_vals) > 0.3:
                        profile['Profile'] = "High Performers 🏆"
                    elif np.mean(avg_vals) < -0.3:
                        profile['Profile'] = "At-Risk Students ⚠️"
                    else:
                        profile['Profile'] = "Average Students 📚"
                    
                    # Add feature-level summary
                    high_features = np.sum(avg_vals > 0.5)
                    low_features = np.sum(avg_vals < -0.5)
                    
                    profile['High Factors'] = high_features
                    profile['Low Factors'] = low_features
                    
                    profiles.append(profile)
                
                st.dataframe(pd.DataFrame(profiles), use_container_width=True)
        
        # Post-Training Correlation Analysis
        st.markdown("---")
        st.markdown("### 📊 Post-Training Feature Analysis")
        st.info("Analyzing correlations between features and **predicted exam scores** from the best regression model.")
        
        # Load metadata to get active features
        metadata = joblib.load("models/metadata.pkl")
        active_features = metadata.get('feature_names', [])
        target_col = metadata.get('target_col', 'Exam_Score')
        
        if st.session_state.df is not None and active_features and eval_results.get('regression'):
            df = st.session_state.df
            
            # Find best regression model
            reg_df = pd.DataFrame(eval_results['regression'])
            best_reg_idx = reg_df['R² Score'].idxmax()
            best_model_key = reg_df.loc[best_reg_idx, 'model_key']
            best_model_name = reg_df.loc[best_reg_idx, 'Model Name']
            
            st.success(f"🎯 Using predictions from: **{best_model_name}** (R² = {reg_df.loc[best_reg_idx, 'R² Score']:.4f})")
            
            # Load the best model and generate predictions
            try:
                best_pipeline = joblib.load(f"models/{best_model_key}.pkl")
                
                # Filter to active features
                df_features = df[active_features].copy()
                
                # Generate predictions for all data
                predicted_scores = best_pipeline.predict(df_features)
                
                # Create analysis dataframe with features + predicted scores
                df_analysis = df_features.copy()
                df_analysis['Predicted_Exam_Score'] = predicted_scores
                
                # Also include actual scores for comparison
                if target_col in df.columns:
                    df_analysis['Actual_Exam_Score'] = df[target_col].values
                
                # Encode categorical features for correlation
                df_encoded = df_analysis.copy()
                for col in df_encoded.columns:
                    if df_encoded[col].dtype == 'object':
                        df_encoded[col] = pd.Categorical(df_encoded[col]).codes
                
                # Correlation Analysis Tabs
                corr_tab1, corr_tab2, corr_tab3 = st.tabs([
                    "🔥 Correlation Heatmap", 
                    "📈 Feature-Prediction Relationships", 
                    "📋 Correlation Table"
                ])
                
                with corr_tab1:
                    st.markdown("#### Correlation Heatmap - Features vs Predicted Scores")
                    st.caption(f"Showing correlations among the {len(active_features)} features and predicted exam scores")
                    
                    # Create correlation matrix
                    corr_matrix = df_encoded.corr()
                    
                    # Create heatmap
                    fig = create_correlation_heatmap(df_encoded, active_features + ['Predicted_Exam_Score'])
                    st.plotly_chart(fig, use_container_width=True, key="post_train_heatmap")
                    
                    # Highlight strongest correlations with predicted scores
                    target_corr = corr_matrix['Predicted_Exam_Score'].drop('Predicted_Exam_Score')
                    if 'Actual_Exam_Score' in target_corr.index:
                        target_corr = target_corr.drop('Actual_Exam_Score')
                    target_corr = target_corr.sort_values(ascending=False)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🔼 Top Positive Correlations with Predicted Score**")
                        top_positive = target_corr.head(5)
                        for feat, corr_val in top_positive.items():
                            st.markdown(f"- **{feat.replace('_', ' ')}**: {corr_val:.3f}")
                    
                    with col2:
                        st.markdown("**🔽 Top Negative Correlations with Predicted Score**")
                        top_negative = target_corr.tail(5)
                        for feat, corr_val in top_negative.items():
                            st.markdown(f"- **{feat.replace('_', ' ')}**: {corr_val:.3f}")
                    
                    # Show prediction accuracy
                    if 'Actual_Exam_Score' in df_encoded.columns:
                        pred_actual_corr = corr_matrix.loc['Predicted_Exam_Score', 'Actual_Exam_Score']
                        st.markdown("---")
                        st.metric(
                            "Prediction Accuracy (Correlation)", 
                            f"{pred_actual_corr:.3f}",
                            help="Correlation between predicted and actual exam scores"
                        )
                
                with corr_tab2:
                    st.markdown("#### Feature vs Predicted Exam Score Relationships")
                    st.caption("Visualize how each feature relates to the model's predictions")
                    
                    # Select feature to visualize
                    selected_feature = st.selectbox(
                        "Select Feature to Analyze",
                        active_features,
                        key="post_train_feature_select"
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Scatter plot with regression line
                        st.markdown(f"**{selected_feature.replace('_', ' ')} vs Predicted Score**")
                        
                        # Check if categorical
                        is_categorical = df[selected_feature].dtype == 'object'
                        
                        if is_categorical:
                            # Use encoded values for plotting but show original labels
                            df_plot = df_encoded.copy()
                            df_plot[selected_feature + '_original'] = df[selected_feature]
                            
                            # Create mapping for x-axis labels
                            unique_vals = df[selected_feature].unique()
                            encoded_vals = pd.Categorical(df[selected_feature]).codes
                            x_tick_map = dict(zip(encoded_vals, unique_vals))
                            
                            fig = create_scatter_with_regression(
                                df_plot, 
                                selected_feature, 
                                'Predicted_Exam_Score',
                                x_tick_map=x_tick_map
                            )
                        else:
                            fig = create_scatter_with_regression(
                                df_encoded, 
                                selected_feature, 
                                'Predicted_Exam_Score'
                            )
                        
                        st.plotly_chart(fig, use_container_width=True, key=f"post_train_scatter_{selected_feature}")
                    
                    with col2:
                        # Distribution comparison
                        st.markdown(f"**{selected_feature.replace('_', ' ')} Distribution**")
                        fig = create_distribution_plot(df, selected_feature)
                        st.plotly_chart(fig, use_container_width=True, key=f"post_train_dist_{selected_feature}")
                    
                    # Show correlation value
                    feature_corr = corr_matrix.loc[selected_feature, 'Predicted_Exam_Score']
                    st.metric(
                        f"Correlation with Predicted Score", 
                        f"{feature_corr:.3f}",
                        help="Pearson correlation coefficient (-1 to 1)"
                    )
                    
                    # Interpretation
                    if abs(feature_corr) > 0.7:
                        strength = "Very Strong"
                    elif abs(feature_corr) > 0.5:
                        strength = "Strong"
                    elif abs(feature_corr) > 0.3:
                        strength = "Moderate"
                    else:
                        strength = "Weak"
                    
                    direction = "Positive" if feature_corr > 0 else "Negative"
                    
                    st.info(f"📊 **{strength} {direction} Correlation**: {selected_feature.replace('_', ' ')} shows a {strength.lower()} {direction.lower()} relationship with predicted exam scores.")
                    
                    # Compare with actual correlation if available
                    if 'Actual_Exam_Score' in df_encoded.columns:
                        actual_corr = corr_matrix.loc[selected_feature, 'Actual_Exam_Score']
                        st.caption(f"ℹ️ Correlation with actual scores: {actual_corr:.3f} (difference: {abs(feature_corr - actual_corr):.3f})")
                
                with corr_tab3:
                    st.markdown("#### Correlation Table - All Active Features")
                    
                    # Create correlation table with predicted scores
                    corr_with_pred = corr_matrix['Predicted_Exam_Score'].drop('Predicted_Exam_Score')
                    if 'Actual_Exam_Score' in corr_with_pred.index:
                        corr_with_pred = corr_with_pred.drop('Actual_Exam_Score')
                    corr_with_pred = corr_with_pred.sort_values(ascending=False)
                    
                    # Also get actual correlations if available
                    if 'Actual_Exam_Score' in df_encoded.columns:
                        corr_with_actual = corr_matrix['Actual_Exam_Score'].drop('Actual_Exam_Score')
                        if 'Predicted_Exam_Score' in corr_with_actual.index:
                            corr_with_actual = corr_with_actual.drop('Predicted_Exam_Score')
                        
                        corr_table = pd.DataFrame({
                            'Feature': [f.replace('_', ' ') for f in corr_with_pred.index],
                            'Predicted Correlation': corr_with_pred.values,
                            'Actual Correlation': [corr_with_actual.get(f, 0) for f in corr_with_pred.index],
                            'Difference': [abs(corr_with_pred[f] - corr_with_actual.get(f, 0)) for f in corr_with_pred.index],
                            'Relationship': ['Positive ↗' if x > 0 else 'Negative ↘' for x in corr_with_pred.values]
                        })
                    else:
                        corr_table = pd.DataFrame({
                            'Feature': [f.replace('_', ' ') for f in corr_with_pred.index],
                            'Predicted Correlation': corr_with_pred.values,
                            'Abs Correlation': abs(corr_with_pred.values),
                            'Relationship': ['Positive ↗' if x > 0 else 'Negative ↘' for x in corr_with_pred.values]
                        })
                    
                    # Style the table
                    def color_correlation(val):
                        if abs(val) > 0.7:
                            color = 'rgba(0, 255, 0, 0.3)'
                        elif abs(val) > 0.5:
                            color = 'rgba(255, 255, 0, 0.3)'
                        elif abs(val) > 0.3:
                            color = 'rgba(255, 165, 0, 0.3)'
                        else:
                            color = ''
                        return f'background-color: {color}'
                    
                    if 'Actual Correlation' in corr_table.columns:
                        st.dataframe(
                            corr_table.style.applymap(color_correlation, subset=['Predicted Correlation', 'Actual Correlation'])
                            .format({
                                'Predicted Correlation': '{:.3f}', 
                                'Actual Correlation': '{:.3f}',
                                'Difference': '{:.3f}'
                            }),
                            use_container_width=True
                        )
                    else:
                        st.dataframe(
                            corr_table.style.applymap(color_correlation, subset=['Predicted Correlation'])
                            .format({'Predicted Correlation': '{:.3f}', 'Abs Correlation': '{:.3f}'}),
                            use_container_width=True
                        )
                    
                    st.caption("""
                    **Color Legend**: 
                    🟢 Green = Very Strong (|r| > 0.7) | 
                    🟡 Yellow = Strong (|r| > 0.5) | 
                    🟠 Orange = Moderate (|r| > 0.3) | 
                    ⚪ None = Weak (|r| ≤ 0.3)
                    """)
                    
                    # Summary statistics
                    st.markdown("#### Correlation Summary")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Strongest Positive", f"{corr_with_pred.max():.3f}")
                        st.caption(corr_with_pred.idxmax().replace('_', ' '))
                    with col2:
                        st.metric("Strongest Negative", f"{corr_with_pred.min():.3f}")
                        st.caption(corr_with_pred.idxmin().replace('_', ' '))
                    with col3:
                        st.metric("Average |Correlation|", f"{abs(corr_with_pred).mean():.3f}")
                        st.caption("Mean absolute correlation")
                    
                    if 'Actual Correlation' in corr_table.columns:
                        st.markdown("---")
                        avg_diff = corr_table['Difference'].mean()
                        st.info(f"📊 **Model Learning Quality**: Average difference between predicted and actual correlations is **{avg_diff:.3f}**. Lower values indicate the model learned the true relationships well.")
            
            except Exception as e:
                st.error(f"Error generating predictions for correlation analysis: {e}")
                import traceback
                st.code(traceback.format_exc())
        
    elif os.path.exists("models/metadata.pkl"):
        st.session_state.models_trained = True
        st.info("Models are trained. Retrain to see detailed evaluation metrics.")
    else:
        st.warning("No models found. Please train models first.")


def render_predictions_page():
    """Render the predictions page using saved models."""
    st.markdown("## 🎯 Make Predictions")
    
    if not os.path.exists("models/metadata.pkl"):
        st.warning("⚠️ No trained models found. Please run the training script.")
        return
        
    try:
        metadata = joblib.load("models/metadata.pkl")
    except Exception as e:
        st.error(f"Error loading model metadata: {e}")
        return

    st.info("Enter student information below to predict performance.")
    
    # Create input form based on trained features
    # We use metadata['feature_names'] to ensure we match the model's expected input
    feature_names = metadata['feature_names']
    categorical_features = metadata.get('categorical_features', [])
    
    if st.session_state.df is None:
        st.warning("Please load the dataset on the Home page first to populate dropdown options.")
        return
    
    # Use the original dataset for reference ranges
    df_full = st.session_state.df
    
    st.markdown("### Choose Model")
    model_type = st.radio("Prediction Type", ["regression", "classification"], horizontal=True, key="pred_type_radio")
    
    model_options = {
        "regression": [("linear_regression", "Linear Regression"), 
                       ("random_forest_regressor", "Random Forest Regressor"), 
                       ("mlp_regressor", "MLP Regressor")],
        "classification": [("logistic_regression", "Logistic Regression"), 
                          ("random_forest_classifier", "Random Forest Classifier"), 
                          ("knn", "K-Nearest Neighbors"), 
                          ("svm", "Support Vector Machine"), 
                          ("mlp_classifier", "MLP Classifier")]
    }
    
    # Display model names but use keys for file paths
    model_display_options = [display for _, display in model_options[model_type]]
    model_keys = [key for key, _ in model_options[model_type]]
    
    selected_display = st.selectbox("Select Model", model_display_options, key=f"model_select_{model_type}")
    selected_model_name = model_keys[model_display_options.index(selected_display)]
    
    st.markdown("---")
    st.markdown("### 📝 Enter Student Data")
    st.caption("💡 You can enter values outside the typical range for exploratory predictions.")
    
    with st.form("prediction_form"):
        input_data = {}
        cols = st.columns(3)
        
        for i, col in enumerate(feature_names):
            with cols[i % 3]:
                if col in categorical_features:
                    # Categorical: Selectbox with all unique values from full dataset
                    if col in df_full.columns:
                        options = sorted(df_full[col].dropna().unique().tolist())
                    else:
                        options = ["Low", "Medium", "High"]
                    input_data[col] = st.selectbox(f"{col.replace('_', ' ')}", options, key=f"input_{col}")
                else:
                    # Numerical: Number input with flexible ranges
                    if col in df_full.columns:
                        # Get statistics from full dataset
                        col_min = float(df_full[col].min())
                        col_max = float(df_full[col].max())
                        col_median = float(df_full[col].median())
                        col_std = float(df_full[col].std())
                        
                        # Set flexible bounds for help text
                        flexible_min = max(0, col_min - 2 * col_std)
                        flexible_max = col_max + 2 * col_std
                        
                        # Determine step size
                        if df_full[col].dtype == 'int64':
                            step = 1.0
                        else:
                            step = 0.1
                    else:
                        # Fallback values if column not in dataset
                        flexible_min = 0.0
                        flexible_max = 100.0
                        col_median = 50.0
                        step = 1.0
                    
                    input_data[col] = st.number_input(
                        f"{col.replace('_', ' ')}", 
                        min_value=None, 
                        max_value=None, 
                        value=col_median,
                        step=step,
                        key=f"input_{col}",
                        help=f"Typical range: {flexible_min:.1f} - {flexible_max:.1f}"
                    )
        
        predict_btn = st.form_submit_button("🚀 Predict Result", type="primary")
        
    if predict_btn:
        try:
            # Load the selected model
            model_path = f"models/{selected_model_name}.pkl"
            if not os.path.exists(model_path):
                st.error(f"Model file not found: {model_path}")
                return
                
            pipeline = joblib.load(model_path)
            
            # Create DataFrame for prediction
            input_df = pd.DataFrame([input_data])
            
            # Predict
            prediction = pipeline.predict(input_df)
            
            st.markdown("---")
            st.markdown("### 🔮 Prediction Result")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**Selected Model:** `{selected_display}`")
                
                if model_type == "regression":
                    score = prediction[0]
                    st.metric("Predicted Exam Score", f"{score:.2f}")
                    
                    # Load metadata to get pass threshold
                    metadata = joblib.load("models/metadata.pkl")
                    pass_thresh = metadata.get('pass_threshold', 60)
                    
                    if score >= pass_thresh:
                        st.success(f"Result: PASS 🎉 (threshold: {pass_thresh}%)")
                    else:
                        st.error(f"Result: FAIL ⚠️ (threshold: {pass_thresh}%)")
                else:
                    result = prediction[0]
                    # Classification returns 0 or 1
                    status = "PASS 🎉" if result == 1 else "FAIL ⚠️"
                    st.metric("Predicted Status", status)
                    
                    # If model supports probability
                    if hasattr(pipeline.named_steps.get('classifier', pipeline.named_steps.get('regressor')), 'predict_proba'):
                        probs = pipeline.predict_proba(input_df)
                        confidence = np.max(probs) * 100
                        st.caption(f"Confidence: {confidence:.1f}%")

            with col2:
                # Show key influencing factors (if linear/logistic)
                st.info("Input Summary")
                st.json(input_data)
            
            # Model Comparison Section
            st.markdown("---")
            st.markdown("### 📊 Model Comparison & Performance")
            
            # Load evaluation results
            if os.path.exists("models/evaluation_results.pkl"):
                eval_results = joblib.load("models/evaluation_results.pkl")
                
                # Create tabs for regression and classification comparisons
                comp_tab1, comp_tab2 = st.tabs(["📉 All Regression Models", "🏷️ All Classification Models"])
                
                with comp_tab1:
                    st.markdown("#### Regression Models - Predictions & Accuracy")
                    
                    reg_predictions = []
                    for model_info in eval_results.get('regression', []):
                        model_key = model_info['model_key']
                        model_name = model_info['Model Name']
                        
                        # Load and predict with this model
                        try:
                            model_pipeline = joblib.load(f"models/{model_key}.pkl")
                            pred = model_pipeline.predict(input_df)[0]
                            
                            reg_predictions.append({
                                'Model': model_name,
                                'Predicted Score': f"{pred:.2f}",
                                'R² Score': f"{model_info['R² Score']:.4f}",
                                'MAE': f"{model_info['MAE']:.2f}",
                                'RMSE': f"{model_info['RMSE']:.2f}"
                            })
                        except:
                            pass
                    
                    if reg_predictions:
                        reg_df = pd.DataFrame(reg_predictions)
                        
                        # Highlight the selected model
                        def highlight_selected(row):
                            if selected_model_name in ['linear_regression', 'random_forest_regressor', 'mlp_regressor']:
                                selected_name = [m['Model Name'] for m in eval_results['regression'] if m['model_key'] == selected_model_name][0]
                                if row['Model'] == selected_name:
                                    return ['background-color: rgba(0, 123, 255, 0.2)'] * len(row)
                            return [''] * len(row)
                        
                        st.dataframe(reg_df.style.apply(highlight_selected, axis=1), use_container_width=True)
                        
                        st.caption("💡 The highlighted row shows your currently selected model. R² closer to 1.0 indicates better accuracy.")
                
                with comp_tab2:
                    st.markdown("#### Classification Models - Predictions & Accuracy")
                    
                    clf_predictions = []
                    for model_info in eval_results.get('classification', []):
                        model_key = model_info['model_key']
                        model_name = model_info['Model Name']
                        
                        # Load and predict with this model
                        try:
                            model_pipeline = joblib.load(f"models/{model_key}.pkl")
                            pred = model_pipeline.predict(input_df)[0]
                            status = "PASS ✅" if pred == 1 else "FAIL ❌"
                            
                            clf_predictions.append({
                                'Model': model_name,
                                'Prediction': status,
                                'Accuracy': f"{model_info['Accuracy']:.4f}",
                                'Precision': f"{model_info['Precision']:.4f}",
                                'Recall': f"{model_info['Recall']:.4f}",
                                'F1 Score': f"{model_info['F1 Score']:.4f}"
                            })
                        except:
                            pass
                    
                    if clf_predictions:
                        clf_df = pd.DataFrame(clf_predictions)
                        
                        # Highlight the selected model
                        def highlight_selected_clf(row):
                            if selected_model_name in ['logistic_regression', 'random_forest_classifier', 'knn', 'svm', 'mlp_classifier']:
                                selected_name = [m['Model Name'] for m in eval_results['classification'] if m['model_key'] == selected_model_name][0]
                                if row['Model'] == selected_name:
                                    return ['background-color: rgba(0, 123, 255, 0.2)'] * len(row)
                            return [''] * len(row)
                        
                        st.dataframe(clf_df.style.apply(highlight_selected_clf, axis=1), use_container_width=True)
                        
                        st.caption("💡 The highlighted row shows your currently selected model. Higher accuracy/F1 scores indicate better performance.")
                
                # Consensus Analysis
                st.markdown("---")
                st.markdown("### 🎯 Model Consensus Analysis")
                
                if model_type == "regression" and reg_predictions:
                    scores = [float(p['Predicted Score']) for p in reg_predictions]
                    avg_score = np.mean(scores)
                    std_score = np.std(scores)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Average Prediction", f"{avg_score:.2f}")
                    with col2:
                        st.metric("Prediction Range", f"±{std_score:.2f}")
                    with col3:
                        agreement = "High" if std_score < 5 else ("Medium" if std_score < 10 else "Low")
                        st.metric("Model Agreement", agreement)
                    
                    st.info(f"📊 All regression models predict an average score of **{avg_score:.1f}** with a standard deviation of **{std_score:.1f}**. Lower deviation indicates higher agreement between models.")
                
                elif model_type == "classification" and clf_predictions:
                    pass_count = sum(1 for p in clf_predictions if "PASS" in p['Prediction'])
                    fail_count = len(clf_predictions) - pass_count
                    consensus = "PASS ✅" if pass_count > fail_count else "FAIL ❌"
                    confidence = max(pass_count, fail_count) / len(clf_predictions) * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Consensus Prediction", consensus)
                    with col2:
                        st.metric("Models Agreeing", f"{max(pass_count, fail_count)}/{len(clf_predictions)}")
                    with col3:
                        st.metric("Consensus Confidence", f"{confidence:.0f}%")
                    
                    st.info(f"📊 **{pass_count}** models predict PASS, **{fail_count}** predict FAIL. The consensus is **{consensus}** with **{confidence:.0f}%** agreement.")
            else:
                st.warning("Model evaluation results not found. Please retrain models to see comparison metrics.")
                
        except Exception as e:
            st.error(f"Prediction Error: {e}")
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
