"""
Student Performance Prediction Dashboard
A comprehensive ML solution with interactive visualizations.

Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_loader import (
    load_data, clean_data, encode_categorical, 
    get_feature_target_split, split_data, scale_features,
    prepare_data_for_classification, get_data_summary, download_dataset
)
from src.feature_engineering import add_all_simulated_features, get_focus_features
from src.models import StudentPerformanceModels, train_all_models
from src.visualizations import (
    create_correlation_heatmap, create_feature_importance_chart,
    create_distribution_plot, create_scatter_with_regression,
    create_box_plot, create_model_comparison_chart,
    create_confusion_matrix_plot, create_cluster_visualization,
    create_loss_curve, create_actual_vs_predicted, create_pie_chart,
    create_metrics_radar_chart
)

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
    st.session_state.selected_features = []


def load_and_prepare_data():
    """Load and prepare the dataset with simulated features."""
    with st.spinner("📥 Downloading dataset from Kaggle..."):
        try:
            df = load_data()
            df = clean_data(df)
            df_enhanced = add_all_simulated_features(df)
            st.session_state.df = df
            st.session_state.df_enhanced = df_enhanced
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
    
    # Auto-initialize if empty
    if not st.session_state.selected_features:
        numeric_df = df_full.select_dtypes(include=[np.number])
        correlations = numeric_df.corr()['Exam_Score'].abs().sort_values(ascending=False)
        top_features = correlations[correlations.index != 'Exam_Score'].head(12).index.tolist()
        st.session_state.selected_features = top_features if top_features else all_cols[:12]

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
             "🤖 Model Training", "🎯 Predictions", "📋 Report"],
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
        st.markdown("### 📁 Datasets")
        st.markdown("- [Student Performance Factors](https://www.kaggle.com/datasets/ayeshaseherr/student-performance)")
        st.markdown("- [Social Media Addiction vs Relationships](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships)")

        if st.session_state.data_loaded:
            st.markdown("---")
            st.info("💡 Configuration Hub now on Home page.")
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
    elif page == "📋 Report":
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
    selected_features = st.session_state.selected_features
    # Always include target
    cols_to_show = selected_features + ['Exam_Score']
    df = df_full[cols_to_show]
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "📈 Statistics", "❓ Missing Values", "🔧 Simulated Features"])
    
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
        st.markdown("### Simulated Features")
        st.info("""
        The following features were simulated to align with the project's focus on 
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
        
        # Show distribution of simulated features
        st.markdown("#### Distribution of Simulated Features")
        
        simulated_cols = ['Social_Media_Hours', 'Movie_Addiction', 'Relationship_Status', 'Gym_Discipline']
        for i, col in enumerate(simulated_cols):
            if col in df.columns:
                if df[col].dtype == 'object':
                    st.plotly_chart(create_pie_chart(df[col], col), width='stretch', key=f"sim_pie_{col}")
                else:
                    st.plotly_chart(create_distribution_plot(df, col), width='stretch', key=f"sim_dist_{col}")
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
            # Only numeric selected features
            numeric_selected = [c for c in selected_features if df_full[c].dtype in ['int64', 'float64', 'int32', 'float32']]
            x_feature = st.selectbox("X-axis Feature", 
                                     numeric_selected if numeric_selected else selected_features[:1],
                                     key="scatter_x")
        with col2:
            # Only categorical selected features
            cat_selected = [c for c in selected_features if df_full[c].dtype == 'object']
            color_by = st.selectbox("Color by (optional)", 
                                    ['None'] + cat_selected,
                                    key="scatter_color")
        
        color_col = None if color_by == 'None' else color_by
        fig = create_scatter_with_regression(df, x_feature, 'Exam_Score', color_col)
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
        st.markdown("### Numerical Focus Factors")
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
    st.markdown("## 🤖 Model Training & Evaluation")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first from the Home page.")
        return
    
    df = st.session_state.df_enhanced
    selected_features = st.session_state.selected_features
    
    if not selected_features:
        st.error("⚠️ No features selected! Please select active factors in the Sidebar.")
        return

    st.markdown("### ⚙️ Training Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05, key="test_size")
    with col2:
        pass_threshold = st.slider("Pass Threshold (for classification)", 50, 70, 60, key="pass_threshold")
    with col3:
        k_neighbors = st.slider("K for KNN", 3, 15, 5, key="k_neighbors")
    
    # Feature Selection summary
    st.markdown("### 🎯 Active Model Features")
    st.info(f"Training models using {len(selected_features)} selected factors. Check the sidebar to add/remove factors.")
    st.write(", ".join(selected_features))
    
    # Train button
    if st.button("🚀 Train All Models", type="primary", width='stretch'):
        with st.spinner("Training models... This may take a moment."):
            try:
                # Prepare data with selected features only
                df_encoded, encoders = encode_categorical(df)
                
                # Filter to only selected features + target
                cols_to_use = selected_features + ['Exam_Score']
                df_filtered = df_encoded[cols_to_use]
                
                X, y = get_feature_target_split(df_filtered, 'Exam_Score')
                X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
                
                # Scale features
                X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
                X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
                X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
                
                # Prepare classification target
                y_train_clf = prepare_data_for_classification(y_train, pass_threshold)
                y_test_clf = prepare_data_for_classification(y_test, pass_threshold)
                
                # Train models
                model_handler = StudentPerformanceModels()
                
                # Regression models
                model_handler.train_linear_regression(X_train_scaled, y_train, X_test_scaled, y_test)
                model_handler.train_random_forest_regressor(X_train_scaled, y_train, X_test_scaled, y_test)
                model_handler.train_mlp_regressor(X_train_scaled, y_train, X_test_scaled, y_test)
                
                # Classification models
                model_handler.train_logistic_regression(X_train_scaled, y_train_clf, X_test_scaled, y_test_clf)
                model_handler.train_random_forest_classifier(X_train_scaled, y_train_clf, X_test_scaled, y_test_clf)
                model_handler.train_knn(X_train_scaled, y_train_clf, X_test_scaled, y_test_clf, k_neighbors)
                model_handler.train_svm(X_train_scaled, y_train_clf, X_test_scaled, y_test_clf)
                model_handler.train_mlp_classifier(X_train_scaled, y_train_clf, X_test_scaled, y_test_clf)
                
                # Clustering
                model_handler.train_kmeans(X_train_scaled, n_clusters=3)
                
                # Store in session state
                st.session_state.model_handler = model_handler
                st.session_state.models_trained = True
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_test_clf = y_test_clf
                st.session_state.scaler = scaler
                st.session_state.encoders = encoders
                st.session_state.feature_columns = X.columns.tolist()
                
                st.success("✅ All models trained successfully!")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error training models: {e}")
                import traceback
                st.code(traceback.format_exc())
    
    # Display results if models are trained
    if st.session_state.models_trained:
        st.markdown("---")
        st.markdown("### 📊 Model Results")
        
        model_handler = st.session_state.model_handler
        
        # Tabs for different model types
        tab1, tab2, tab3 = st.tabs(["📉 Regression Models", "🏷️ Classification Models", "🔮 Clustering"])
        
        with tab1:
            st.markdown("#### Regression Model Comparison")
            st.info("The table below compares regression models across key metrics. The best performing model for each metric is highlighted in green.")
            
            reg_comparison = model_handler.get_model_comparison('regression')
            if not reg_comparison.empty:
                # Transpose for easier row-wise comparison of models
                reg_comp_t = reg_comparison.set_index('Model Name').T
                
                def highlight_best_reg_t(row):
                    is_min_metrics = ['MAE', 'MSE', 'RMSE']
                    if row.name in is_min_metrics:
                        best_val = row.min()
                    else:
                        best_val = row.max()
                    return ['background-color: rgba(0, 255, 0, 0.3)' if v == best_val else '' for v in row]

                st.dataframe(reg_comp_t.style.apply(highlight_best_reg_t, axis=1).format(precision=4), width='stretch')
                
                # Visualization
                col1, col2 = st.columns(2)
                with col1:
                    fig = create_model_comparison_chart(reg_comparison, 'R² Score')
                    st.plotly_chart(fig, width='stretch', key="reg_r2_comp")
                with col2:
                    fig = create_model_comparison_chart(reg_comparison, 'MAE')
                    st.plotly_chart(fig, width='stretch', key="reg_mae_comp")
                
                # Actual vs Predicted for Linear Regression
                if 'linear_regression' in model_handler.results:
                    st.markdown("#### Linear Regression: Actual vs Predicted")
                    lr_results = model_handler.results['linear_regression']
                    fig = create_actual_vs_predicted(
                        st.session_state.y_test.values,
                        lr_results['predictions']
                    )
                    st.plotly_chart(fig, width='stretch', key="lr_actual_pred")
                    
                    # Feature importance
                    st.markdown("#### Feature Importance (Linear Regression Coefficients)")
                    coefficients = lr_results.get('coefficients', {})
                    if coefficients:
                        fig = create_feature_importance_chart(coefficients)
                        st.plotly_chart(fig, width='stretch', key="lr_feat_imp")
        
        with tab2:
            st.markdown("#### Classification Model Comparison")
            st.info("The table below compares classification models. The best performing model for each metric is highlighted in green.")
            
            clf_comparison = model_handler.get_model_comparison('classification')
            if not clf_comparison.empty:
                # Transpose for easier row-wise comparison
                clf_comp_t = clf_comparison.set_index('Model Name').T
                
                def highlight_best_clf_t(row):
                    # For classification metrics, higher is always better
                    best_val = row.max()
                    return ['background-color: rgba(0, 255, 0, 0.3)' if v == best_val else '' for v in row]

                st.dataframe(clf_comp_t.style.apply(highlight_best_clf_t, axis=1).format(precision=4), width='stretch')
                
                # Visualizations
                col1, col2 = st.columns(2)
                with col1:
                    fig = create_model_comparison_chart(clf_comparison, 'Accuracy')
                    st.plotly_chart(fig, width='stretch', key="clf_acc_comp")
                with col2:
                    fig = create_model_comparison_chart(clf_comparison, 'F1 Score')
                    st.plotly_chart(fig, width='stretch', key="clf_f1_comp")
                
                # Radar chart
                st.markdown("#### Model Comparison Radar Chart")
                fig = create_metrics_radar_chart(clf_comparison)
                if fig:
                    st.plotly_chart(fig, width='stretch', key="clf_radar")
                
                # Confusion matrices
                st.markdown("#### Confusion Matrices")
                clf_models_to_show = [m for m in ['logistic_regression', 'random_forest_classifier', 'knn', 'svm', 'mlp_classifier'] if m in model_handler.results]
                
                # Show in a grid (2 per row for better visibility)
                num_cols = 2
                for row_idx in range((len(clf_models_to_show) + num_cols - 1) // num_cols):
                    cols = st.columns(num_cols)
                    for col_idx in range(num_cols):
                        model_idx = row_idx * num_cols + col_idx
                        if model_idx < len(clf_models_to_show):
                            m_key = clf_models_to_show[model_idx]
                            with cols[col_idx]:
                                m_res = model_handler.results[m_key]
                                st.markdown(f"**{m_res['model_name']}**")
                                fig = create_confusion_matrix_plot(m_res['confusion_matrix'])
                                st.plotly_chart(fig, width='stretch', key=f"cm_{m_key}")
                
                # MLP Loss curve
                if 'mlp_classifier' in model_handler.results:
                    loss_curve = model_handler.results['mlp_classifier'].get('loss_curve')
                    if loss_curve:
                        st.markdown("#### Neural Network Training Loss")
                        fig = create_loss_curve(loss_curve)
                        st.plotly_chart(fig, width='stretch', key="mlp_loss")
        
        with tab3:
            st.markdown("#### K-Means Clustering Results")
            
            if 'kmeans' in model_handler.results:
                km_results = model_handler.results['kmeans']
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Number of Clusters", km_results['n_clusters'])
                with col2:
                    st.metric("Silhouette Score", f"{km_results['silhouette_score']:.3f}")
                with col3:
                    st.metric("Inertia", f"{km_results['inertia']:.1f}")
                
                st.markdown("#### Cluster Profiles & Interpretations")
                
                centers = km_results['cluster_centers']
                features = km_results['feature_names']
                
                if features:
                    profiles = []
                    for i in range(len(centers)):
                        profile = {'Cluster': f"Group {i}"}
                        # Create descriptive names based on key features
                        score_idx = features.index('Exam_Score') if 'Exam_Score' in features else -1
                        study_idx = features.index('Hours_Studied') if 'Hours_Studied' in features else -1
                        health_idx = features.index('mental_health_rating') if 'mental_health_rating' in features else -1
                        
                        # Interpretation logic
                        avg_score = centers[i][score_idx] if score_idx != -1 else 0
                        avg_study = centers[i][study_idx] if study_idx != -1 else 0
                        
                        group_name = "Standard Students"
                        if avg_score > 0.5 and avg_study > 0.5: group_name = "Academic Achievers 🏆"
                        elif avg_score < -0.5: group_name = "At-Risk Students ⚠️"
                        elif avg_study > 0.5 and avg_score < 0: group_name = "Hard Workers (Low Efficiency) 📚"
                        elif avg_study < -0.5 and avg_score > 0: group_name = "High Efficiency Students ⚡"
                        
                        profile['Group Name'] = group_name
                        profile['Size'] = km_results['cluster_sizes'].get(i, 0)
                        
                        # Add a few representative feature levels
                        for feat_idx, feat_name in enumerate(features[:5]): # Show first 5 features
                            val = centers[i][feat_idx]
                            profile[feat_name] = "High (+) " if val > 0.3 else ("Low (-)" if val < -0.3 else "Average")
                            
                        profiles.append(profile)
                    
                    st.dataframe(pd.DataFrame(profiles), width='stretch')
                else:
                    st.dataframe(cluster_sizes, width='stretch')


def render_predictions_page():
    """Render the predictions page."""
    st.markdown("## 🎯 Make Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first from the Model Training page.")
        return
    
    st.info("Enter student information to predict their exam score and pass/fail status.")
    
    df_full = st.session_state.df_enhanced
    selected_features = st.session_state.selected_features
    df = df_full[selected_features + ['Exam_Score']]
    model_handler = st.session_state.model_handler
    
    def get_opts(col, default_opts):
        if col in df.columns and df[col].dtype == 'object':
            return sorted(df[col].unique().tolist())
        return default_opts

    # Dynamic input form based on selected features
    st.markdown("### 📝 Enter Student Data")
    st.caption("Only 'Active Factors' selected on the Home page are shown here.")
    
    input_data = {}
    
    # Configuration for widgets
    # Format: label, type (slider/select), min, max, default, step/options
    config = {
        'Hours_Studied': ("📚 Hours Studied", "slider", 0.0, 15.0, 5.0, 0.5),
        'Attendance': ("📅 Attendance (%)", "slider", 0, 100, 85, 1),
        'Previous_Scores': ("📉 Previous GPA", "slider", 0.0, 4.0, 3.2, 0.1),
        'exam_anxiety_score': ("🧠 Exam Anxiety", "slider", 0, 100, 30, 1),
        'Sleep_Hours': ("😴 Sleep Hours", "slider", 0.0, 12.0, 7.0, 0.5),
        'Social_Media_Hours': ("📱 Social Media Hours", "slider", 0.0, 15.0, 2.0, 0.5),
        'Movie_Hours': ("🎬 Movie/Netflix Hours", "slider", 0.0, 10.0, 1.5, 0.5),
        'Physical_Activity': ("🏋️ Physical Activity", "slider", 0, 20, 3, 1),
        'mental_health_rating': ("🧘 Mental Health", "slider", 1, 5, 4, 1),
        'stress_level': ("😰 Stress Level", "slider", 1, 5, 2, 1),
        'Tutoring_Sessions': ("🏫 Tutoring Sessions", "slider", 0, 15, 2, 1),
        'Age': ("🎂 Age", "slider", 15, 60, 20, 1),
        'age': ("🎂 Age", "slider", 15, 60, 20, 1),
        'Motivation_Level': ("🔥 Motivation Level", "selectbox", ["Low", "Medium", "High"]),
        'Internet_Access': ("🌐 Internet Access", "selectbox", ["Yes", "No"]),
        'Gender': ("👤 Gender", "selectbox", ["Male", "Female"]),
        'Relationship_Status': ("💑 Relationship Status", "selectbox", ["Single", "In Relationship"]),
        'Family_Income': ("💰 Family Income", "selectbox", ["Low", "Medium", "High"]),
        'Teacher_Quality': ("👨‍🏫 Teacher Quality", "selectbox", ["Low", "Medium", "High"]),
        'School_Type': ("🏫 School Type", "selectbox", ["Public", "Private"]),
        'Parental_Involvement': ("👪 Parental Involvement", "selectbox", ["Low", "Medium", "High"]),
        'Access_to_Resources': ("📚 Access to Resources", "selectbox", ["Low", "Medium", "High"]),
        'Parental_Education_Level': ("🎓 Parental Education", "selectbox", ["High School", "College", "Postgraduate"]),
        'major': ("🎓 Major", "selectbox", ["Computer Science", "Engineering", "Arts", "Business"]),
        'Extracurricular_Activities': ("⚽ Extracurriculars", "selectbox", ["Yes", "No"]),
        'Peer_Influence': ("👥 Peer Influence", "selectbox", ["Positive", "Neutral", "Negative"]),
        'Learning_Disabilities': ("⚠️ Learning Disabilities", "selectbox", ["No", "Yes"]),
        'Distance_from_Home': ("🏠 Distance from Home", "selectbox", ["Near", "Moderate", "Far"]),
    }

    # Split into 3 columns
    cols = st.columns(3)
    for i, field in enumerate(selected_features):
        with cols[i % 3]:
            if field in config:
                info = config[field]
                if info[1] == "slider":
                    input_data[field] = st.slider(info[0], info[2], info[3], info[4], info[5])
                else:
                    input_data[field] = st.selectbox(info[0], get_opts(field, info[2]))
            else:
                # Fallback for unexpected features
                if df_full[field].dtype in ['int64', 'float64']:
                    input_data[field] = st.number_input(f"{field}", value=0.0)
                else:
                    input_data[field] = st.selectbox(f"{field}", get_opts(field, ["N/A"]))

    # Prediction button
    if st.button("🔮 Predict Performance", type="primary", width='stretch'):
        try:
            # Handle special cases for internal logic if features were dropped
            if 'Movie_Hours' in input_data:
                mh = input_data['Movie_Hours']
                input_data['Movie_Addiction'] = 'Low' if mh < 1.5 else ('Medium' if mh < 3.5 else 'High')
            
            # Create safety copy of input_data for dataframe
            final_input = input_data.copy()
            
            # Ensure every single column the model was trained on exists (for safety)
            feature_columns = st.session_state.feature_columns
            for col in feature_columns:
                if col not in final_input:
                    # Fill dropped features with 0 (numeric) or a default string
                    if df_full[col].dtype in ['int64', 'float64']:
                        final_input[col] = 0
                    else:
                        # Try to get first unique value or "N/A"
                        final_input[col] = str(df_full[col].mode()[0]) if col in df_full.columns else "N/A"
            
            input_df = pd.DataFrame([final_input])
            
            # CRITICAL: Ensure every single column the model was trained on exists here
            feature_columns = st.session_state.feature_columns
            
            # Encode categorical variables using the saved encoders
            encoders = st.session_state.encoders
            feature_columns = st.session_state.feature_columns
            
            # Ensure every column that was an object during training is encoded here
            for col in input_df.columns:
                if col in encoders:
                    try:
                        # Convert value to string to match fit_transform format
                        val = str(input_df.at[0, col])
                        # transform expects a sequence-like object
                        input_df[col] = encoders[col].transform([val])[0]
                    except Exception as e:
                        # Fallback for unseen labels: use 0 or common label
                        input_df[col] = 0
            
            # Ensure all numeric columns are actually numeric (not strings)
            # This handles any cases where encoding might have been skipped or failed
            for col in input_df.columns:
                if col not in encoders:
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0)
            
            # Ensure column order matches training EXACTLY
            input_df = input_df[feature_columns]
            
            # Scale
            scaler = st.session_state.scaler
            input_scaled_arr = scaler.transform(input_df)
            input_scaled = pd.DataFrame(input_scaled_arr, columns=feature_columns)
            
            # 1. Prediction Comparison Table (separately for all models)
            st.markdown("### 📊 Model Comparison for this Prediction")
            
            reg_models = {
                'linear_regression': 'Linear Regression',
                'random_forest_regressor': 'Random Forest',
                'mlp_regressor': 'Neural Network (MLP)'
            }
            
            clf_models = {
                'logistic_regression': 'Logistic Regression',
                'random_forest_classifier': 'Random Forest',
                'knn': 'K-Nearest Neighbors',
                'svm': 'SVM',
                'mlp_classifier': 'Neural Network (MLP)'
            }
            
            # Collect all predictions
            all_preds = []
            for m_key, m_name in reg_models.items():
                if m_key in model_handler.models:
                    pred = model_handler.predict(m_key, input_scaled)[0]
                    reliability = model_handler.results[m_key]['test_r2'] * 100
                    all_preds.append({
                        'Type': 'Score Prediction',
                        'Model': m_name,
                        'Prediction': f"{pred:.1f}",
                        'Reliability (Accuracy)': f"{max(0, reliability):.1f}%"
                    })
            
            for m_key, m_name in clf_models.items():
                if m_key in model_handler.models:
                    pred = model_handler.predict(m_key, input_scaled)[0]
                    precision = model_handler.results[m_key]['precision'] * 100
                    all_preds.append({
                        'Type': 'Pass/Fail Status',
                        'Model': m_name,
                        'Prediction': "Pass ✅" if pred == 1 else "Fail ❌",
                        'Reliability (Precision)': f"{precision:.1f}%"
                    })
            
            # Model comparison for this prediction
            st.markdown("### 📊 Model Consensus")
            all_preds_df = pd.DataFrame(all_preds)
            st.dataframe(all_preds_df, width='stretch')
            
            st.markdown("---")
            
            # Detailed View based on selection
            st.markdown("### 🎯 Detailed Prediction Analysis")
            col_sel1, col_sel2 = st.columns(2)
            with col_sel1:
                # Filter reg_models to only those trained
                avail_regs = {k: v for k, v in reg_models.items() if k in model_handler.models}
                selected_reg_name = st.selectbox("Select Regressor for Score", list(avail_regs.values()), key="sel_reg")
            with col_sel2:
                # Filter clf_models to only those trained
                avail_clfs = {k: v for k, v in clf_models.items() if k in model_handler.models}
                selected_clf_name = st.selectbox("Select Classifier for Status", list(avail_clfs.values()), key="sel_clf")
            
            reg_key = [k for k, v in avail_regs.items() if v == selected_reg_name][0]
            clf_key = [k for k, v in avail_clfs.items() if v == selected_clf_name][0]
            
            score_pred = model_handler.predict(reg_key, input_scaled)[0]
            pass_pred = model_handler.predict(clf_key, input_scaled)[0]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Exam Score", f"{score_pred:.1f}")
                rel = model_handler.results[reg_key]['test_r2'] * 100
                st.caption(f"📊 Accuracy ({selected_reg_name}): {max(0, rel):.1f}%")
            
            with col2:
                status = "Pass ✅" if pass_pred == 1 else "Fail ❌"
                st.metric("Pass/Fail Prediction", status)
                prec = model_handler.results[clf_key]['precision'] * 100
                st.caption(f"🎯 Precision ({selected_clf_name}): {prec:.1f}%")
            
            with col3:
                # Grade logic
                g = "F"
                if score_pred >= 90: g = "A"
                elif score_pred >= 80: g = "B"
                elif score_pred >= 70: g = "C"
                elif score_pred >= 60: g = "D"
                st.metric("Predicted Grade", g)
                st.caption("Based on standard GPA scale")

            # Insights summary
            st.markdown("### 💡 Strategic Insights")
            insights = []
            if 'Hours_Studied' in input_data and input_data['Hours_Studied'] < 3:
                insights.append("⚠️ Low study hours detected. Increasing focus blocks could boost your score.")
            if 'Attendance' in input_data and input_data['Attendance'] < 75:
                insights.append("📅 Low attendance is a high-risk factor for exam failure.")
            if 'Sleep_Hours' in input_data and input_data['Sleep_Hours'] < 6:
                insights.append("😴 Lack of sleep significantly impairs cognitive recall during exams.")
            
            if insights:
                for ins in insights: st.write(f"- {ins}")
            else:
                st.success("✨ Good habit balance detected! Keep it up.")

        except Exception as e:
            st.error(f"Prediction Error: {e}")
            import traceback
            st.code(traceback.format_exc())
def render_report_page():
    """Render the final report page."""
    st.markdown("## 📋 Project Report")
    
    st.markdown("""
    ### 1. Introduction
    
    This project explores the relationship between academic performance and various lifestyle/behavioral factors. 
    We leverage Machine Learning to identify the strongest predictors of student success, moving beyond traditional 
    metrics to include mental health, social media habits, and personal relationships.
    
    ### 2. Datasets
    
    We merged two high-quality datasets to provide a holistic view:
    - **[Student Performance Factors](https://www.kaggle.com/datasets/ayeshaseherr/student-performance)**: 10,000 records of academic and demographic data.
    - **[Social Media Addiction vs Relationships](https://www.kaggle.com/datasets/adilshamim8/social-media-addiction-vs-relationships)**: Detailed metrics on digital behavior and its social/psychological impact.
    
    ### 3. Methodology
    
    #### Data Preprocessing
    - **Merged Schema**: Statistically integrated features from both sources.
    - **Handling Missingness**: Implemented median/mode imputation for continuous/categorical data.
    - **Feature Engineering**: Derived movie usage proxy and gym discipline categories.
    - **Encodings**: Applied Label Encoding for categorical features and StandardScaler for normalization.
    
    #### Models Implemented
    
    | Model | Type | Purpose |
    |-------|------|---------|
    | Linear Regression | Regression | Predict exact GPA/Exam scores |
    | Random Forest | Ens. Learning | High-performance non-linear modeling |
    | Logistic Regression | Classification | Pass/Fail binary classification |
    | KNN | Classification | Distance-based student grouping |
    | SVM | Classification | High-dimensional classification |
    | K-Means | Clustering | Unsupervised behavioral segmentation |
    | Neural Network (MLP) | Deep Learning | Complex pattern recognition |
    
    ### 4. Key Findings
    """)
    
    if st.session_state.models_trained:
        model_handler = st.session_state.model_handler
        
        # Best models
        if 'linear_regression' in model_handler.results:
            lr_r2 = model_handler.results['linear_regression']['test_r2']
            st.markdown(f"- **Linear Regression R² Score**: {lr_r2:.4f}")
        
        clf_models = ['logistic_regression', 'random_forest_classifier', 'knn', 'svm', 'mlp_classifier']
        best_acc = 0
        best_model = ""
        for model_name in clf_models:
            if model_name in model_handler.results:
                acc = model_handler.results[model_name]['test_accuracy']
                if acc > best_acc:
                    best_acc = acc
                    best_model = model_handler.results[model_name]['model_name']
        
        if best_model:
            st.markdown(f"- **Best Classification Model**: {best_model} (Accuracy: {best_acc:.4f})")
        
        # Top features
        if 'linear_regression' in model_handler.results:
            coefficients = model_handler.results['linear_regression'].get('coefficients', {})
            if coefficients:
                top_features = sorted(coefficients.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
                st.markdown("- **Top 5 Most Important Features**:")
                for feat, coef in top_features:
                    direction = "positive" if coef > 0 else "negative"
                    st.markdown(f"  - {feat}: {direction} impact ({coef:.4f})")
    else:
        st.info("Train models to see detailed findings here.")
    
    st.markdown("""
    ### 5. Conclusions
    
    1. **Previous Academic Performance** (GPA) remains the strongest predictor of future exam scores.
    
    2. **Mental Health and Stress** are critical factors - higher stress levels and anxiety show strong negative correlations with performance.
    
    3. **Social media and Netflix** (entertainment) usage beyond 3-4 hours per day correlates with declining academic scores.
    
    4. **Sleep and Personal Habits** like diet quality and regular exercise (Gym Discipline) have a significant positive impact on cognitive performance.
    
    5. Machine learning models can effectively predict student performance with reasonable accuracy.
    
    ### 6. Future Improvements
    
    - Collect real data on social media usage and movie watching habits
    - Implement more advanced deep learning models
    - Add time-series analysis for tracking performance over time
    - Create personalized recommendations for students based on their profiles
    
    ### 7. References
    
    - Dataset: [Kaggle - Student Performance Factors](https://www.kaggle.com/datasets/ayeshaseherr/student-performance)
    - Libraries: scikit-learn, Streamlit, Pandas, Plotly
    """)


if __name__ == "__main__":
    main()
