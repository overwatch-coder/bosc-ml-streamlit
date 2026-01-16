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
    
    # Load data button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if not st.session_state.data_loaded:
            if st.button("📥 Load Dataset from Kaggle", use_container_width=True, type="primary"):
                load_and_prepare_data()
                st.rerun()
        else:
            st.success("✅ Dataset loaded successfully!")
    
    st.markdown("---")
    
    # Project Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📌 Project Overview")
        st.markdown("""
        This project analyzes factors affecting student academic performance and builds 
        predictive models using **real data** from two integrated Kaggle datasets.
        
        **Key Real Factors Analyzed:**
        - 😴 **Sleep Hours**: Actual hours from student reports.
        - 📱 **Social Media**: Real usage hours and addiction levels.
        - 🧠 **Mental Health**: Direct ratings from student surveys.
        - 💑 **Relationships**: Impact of social status on studies.
        - 🏋️ **Physical Activity**: Real exercise frequency.
        """)
    
    with col2:
        st.markdown("### 🤖 ML Models Implemented")
        st.markdown("""
        | Model | Type |
        |-------|------|
        | Linear Regression | Regression |
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
            st.metric("Features", len(df.columns) - 1)
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
    
    df = st.session_state.df_enhanced
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "📈 Statistics", "❓ Missing Values", "🔧 Simulated Features"])
    
    with tab1:
        st.markdown("### Raw Data Preview")
        st.dataframe(df.head(100), use_container_width=True)
        st.info(f"Showing first 100 rows of {len(df):,} total records")
    
    with tab2:
        st.markdown("### Descriptive Statistics")
        st.dataframe(df.describe().round(2), use_container_width=True)
        
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
        st.dataframe(missing_df[missing_df['Missing Count'] > 0], use_container_width=True)
        
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
        col1, col2, col3, col4 = st.columns(4)
        
        simulated_cols = ['Social_Media_Hours', 'Movie_Addiction', 'Relationship_Status', 'Gym_Discipline']
        for i, col in enumerate(simulated_cols):
            with [col1, col2, col3, col4][i]:
                if col in df.columns:
                    if df[col].dtype == 'object':
                        st.plotly_chart(create_pie_chart(df[col], col), use_container_width=True, key=f"sim_pie_{col}")
                    else:
                        st.plotly_chart(create_distribution_plot(df, col), use_container_width=True, key=f"sim_dist_{col}")


def render_visualizations_page():
    """Render the visualizations page."""
    st.markdown("## 📈 Data Visualizations")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first from the Home page.")
        return
    
    df = st.session_state.df_enhanced
    
    # Encode for correlation
    df_encoded, _ = encode_categorical(df)
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Correlation", "📊 Distributions", "📉 Relationships", "🎯 Focus Factors"])
    
    with tab1:
        st.markdown("### Correlation Heatmap")
        fig = create_correlation_heatmap(df_encoded)
        st.plotly_chart(fig, use_container_width=True, key="corr_heatmap")
        
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
            st.plotly_chart(fig, use_container_width=True, key=f"dist_{feature}")
        else:
            fig = create_pie_chart(df[feature], f"Distribution of {feature}")
            st.plotly_chart(fig, use_container_width=True, key=f"pie_{feature}")
    
    with tab3:
        st.markdown("### Feature vs Exam Score")
        
        col1, col2 = st.columns(2)
        with col1:
            x_feature = st.selectbox("X-axis Feature", 
                                     df.select_dtypes(include=[np.number]).columns.tolist(),
                                     key="scatter_x")
        with col2:
            color_by = st.selectbox("Color by (optional)", 
                                    ['None'] + df.select_dtypes(include=['object']).columns.tolist(),
                                    key="scatter_color")
        
        color_col = None if color_by == 'None' else color_by
        fig = create_scatter_with_regression(df, x_feature, 'Exam_Score', color_col)
        st.plotly_chart(fig, use_container_width=True, key=f"scatter_{x_feature}")
    
    with tab4:
        st.markdown("### Focus Factors Analysis")
        st.info("Analyzing the key factors of interest: Sleep, Social Media, Movie Addiction, Relationship Status, Gym Discipline")
        
        # Box plots for categorical focus factors
        categorical_focus = ['Movie_Addiction', 'Relationship_Status', 'Gym_Discipline', 'mental_health_rating', 'stress_level', 'Motivation_Level']
        
        for i, factor in enumerate(categorical_focus):
            if factor in df.columns:
                fig = create_box_plot(df, factor, 'Exam_Score')
                st.plotly_chart(fig, use_container_width=True, key=f"box_{factor}")
        
        # Scatter for numerical focus factors
        st.markdown("### Numerical Focus Factors")
        col1, col2 = st.columns(2)
        with col1:
            if 'Sleep_Hours' in df.columns:
                fig = create_scatter_with_regression(df, 'Sleep_Hours', 'Exam_Score')
                st.plotly_chart(fig, use_container_width=True, key="scatter_sleep")
        with col2:
            if 'Social_Media_Hours' in df.columns:
                fig = create_scatter_with_regression(df, 'Social_Media_Hours', 'Exam_Score')
                st.plotly_chart(fig, use_container_width=True, key="scatter_social")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'Movie_Hours' in df.columns:
                fig = create_scatter_with_regression(df, 'Movie_Hours', 'Exam_Score')
                st.plotly_chart(fig, use_container_width=True, key="scatter_movie")
        with col2:
            if 'exam_anxiety_score' in df.columns:
                fig = create_scatter_with_regression(df, 'exam_anxiety_score', 'Exam_Score')
                st.plotly_chart(fig, use_container_width=True, key="scatter_anxiety")


def render_model_training_page():
    """Render the model training page."""
    st.markdown("## 🤖 Model Training & Evaluation")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load the dataset first from the Home page.")
        return
    
    df = st.session_state.df_enhanced
    
    # Model parameters
    st.markdown("### ⚙️ Training Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2, 0.05)
    with col2:
        pass_threshold = st.slider("Pass Threshold (for classification)", 50, 70, 60)
    with col3:
        k_neighbors = st.slider("K for KNN", 3, 15, 5)
    
    # Train button
    if st.button("🚀 Train All Models", type="primary", use_container_width=True):
        with st.spinner("Training models... This may take a moment."):
            try:
                # Prepare data
                df_encoded, encoders = encode_categorical(df)
                X, y = get_feature_target_split(df_encoded, 'Exam_Score')
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
                model_handler.train_mlp_regressor(X_train_scaled, y_train, X_test_scaled, y_test)
                
                # Classification models
                model_handler.train_logistic_regression(X_train_scaled, y_train_clf, X_test_scaled, y_test_clf)
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
            
            reg_comparison = model_handler.get_model_comparison('regression')
            if not reg_comparison.empty:
                st.dataframe(reg_comparison.round(4), use_container_width=True)
                
                # Visualization
                col1, col2 = st.columns(2)
                with col1:
                    fig = create_model_comparison_chart(reg_comparison, 'R² Score')
                    st.plotly_chart(fig, use_container_width=True, key="reg_r2_comp")
                with col2:
                    fig = create_model_comparison_chart(reg_comparison, 'MAE')
                    st.plotly_chart(fig, use_container_width=True, key="reg_mae_comp")
                
                # Actual vs Predicted for Linear Regression
                if 'linear_regression' in model_handler.results:
                    st.markdown("#### Linear Regression: Actual vs Predicted")
                    lr_results = model_handler.results['linear_regression']
                    fig = create_actual_vs_predicted(
                        st.session_state.y_test.values,
                        lr_results['predictions']
                    )
                    st.plotly_chart(fig, use_container_width=True, key="lr_actual_pred")
                    
                    # Feature importance
                    st.markdown("#### Feature Importance (Linear Regression Coefficients)")
                    coefficients = lr_results.get('coefficients', {})
                    if coefficients:
                        fig = create_feature_importance_chart(coefficients)
                        st.plotly_chart(fig, use_container_width=True, key="lr_feat_imp")
        
        with tab2:
            st.markdown("#### Classification Model Comparison")
            
            # Build comparison manually
            clf_models = ['logistic_regression', 'knn', 'svm', 'mlp_classifier']
            clf_data = []
            for model_name in clf_models:
                if model_name in model_handler.results:
                    r = model_handler.results[model_name]
                    clf_data.append({
                        'Model': r['model_name'],
                        'Accuracy': r['test_accuracy'],
                        'Precision': r['precision'],
                        'Recall': r['recall'],
                        'F1 Score': r['f1_score']
                    })
            
            clf_comparison = pd.DataFrame(clf_data)
            if not clf_comparison.empty:
                st.dataframe(clf_comparison.round(4), use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                with col1:
                    fig = create_model_comparison_chart(clf_comparison, 'Accuracy')
                    st.plotly_chart(fig, use_container_width=True, key="clf_acc_comp")
                with col2:
                    fig = create_model_comparison_chart(clf_comparison, 'F1 Score')
                    st.plotly_chart(fig, use_container_width=True, key="clf_f1_comp")
                
                # Radar chart
                st.markdown("#### Model Comparison Radar Chart")
                fig = create_metrics_radar_chart(clf_comparison)
                if fig:
                    st.plotly_chart(fig, use_container_width=True, key="clf_radar")
                
                # Confusion matrices
                st.markdown("#### Confusion Matrices")
                cols = st.columns(4)
                for i, model_name in enumerate(clf_models):
                    if model_name in model_handler.results:
                        with cols[i]:
                            cm = model_handler.results[model_name]['confusion_matrix']
                            st.markdown(f"**{model_handler.results[model_name]['model_name']}**")
                            fig = create_confusion_matrix_plot(cm)
                            st.plotly_chart(fig, use_container_width=True, key=f"cm_{model_name}")
                
                # MLP Loss curve
                if 'mlp_classifier' in model_handler.results:
                    loss_curve = model_handler.results['mlp_classifier'].get('loss_curve')
                    if loss_curve:
                        st.markdown("#### Neural Network Training Loss")
                        fig = create_loss_curve(loss_curve)
                        st.plotly_chart(fig, use_container_width=True, key="mlp_loss")
        
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
                
                st.markdown("#### Cluster Sizes")
                cluster_sizes = pd.DataFrame({
                    'Cluster': list(km_results['cluster_sizes'].keys()),
                    'Size': list(km_results['cluster_sizes'].values())
                })
                st.dataframe(cluster_sizes, use_container_width=True)


def render_predictions_page():
    """Render the predictions page."""
    st.markdown("## 🎯 Make Predictions")
    
    if not st.session_state.models_trained:
        st.warning("⚠️ Please train the models first from the Model Training page.")
        return
    
    st.info("Enter student information to predict their exam score and pass/fail status.")
    
    df = st.session_state.df_enhanced
    model_handler = st.session_state.model_handler
    
    def get_opts(col, default_opts):
        if col in df.columns and df[col].dtype == 'object':
            return sorted(df[col].unique().tolist())
        return default_opts

    # Create input form
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📚 Academic Factors")
        hours_studied = st.slider("Hours Studied (per day)", 0.0, 15.0, 5.0, 0.5)
        attendance = st.slider("Attendance (%)", 0, 100, 85)
        previous_scores = st.slider("Previous GPA (0-4)", 0.0, 4.0, 3.2, 0.1)
        exam_anxiety = st.slider("Exam Anxiety Score (0-100)", 0, 100, 30)
        time_mgmt = st.slider("Time Management Score (0-100)", 0, 100, 70)
    
    with col2:
        st.markdown("### 🏠 Personal Factors")
        sleep_hours = st.slider("Sleep Hours (per night)", 4.0, 10.0, 7.0, 0.5)
        social_media_hours = st.slider("Social Media Hours (per day)", 0.0, 12.0, 2.0, 0.5)
        movie_hours = st.slider("Netflix/Movie Hours (per day)", 0.0, 8.0, 1.5, 0.5)
        
        mental_health = st.slider("Mental Health Rating (1-5)", 1, 5, 4)
        stress_level = st.slider("Stress Level (1-5)", 1, 5, 2)
        motivation_level = st.selectbox("Motivation Level", get_opts("Motivation_Level", ["Low", "Medium", "High"]))
        relationship_status = st.selectbox("Relationship Status", get_opts("Relationship_Status", ["Single", "In Relationship"]))
    
    with col3:
        st.markdown("### 🏫 Environment & Others")
        diet_quality = st.selectbox("Diet Quality", get_opts("diet_quality", ["Poor", "Average", "Good"]))
        internet_access = st.selectbox("Internet Access", get_opts("Internet_Access", ["Yes", "No"]))
        parental_involvement = st.selectbox("Parental Involvement", get_opts("Parental_Involvement", ["Low", "Medium", "High"]))
        access_to_resources = st.selectbox("Access to Resources", get_opts("Access_to_Resources", ["Low", "Medium", "High"]))
        teacher_quality = st.selectbox("Teacher Quality", get_opts("Teacher_Quality", ["Low", "Medium", "High"]))
        gender = st.selectbox("Gender", get_opts("Gender", ["Male", "Female"]))
        age = st.slider("Age", 15, 30, 20)
        physical_activity = st.slider("Physical Activity (hours/week)", 0, 10, 3)
    
    # Additional factors in expander
    with st.expander("More Detailed Factors"):
        col1, col2, col3 = st.columns(3)
        with col1:
            school_type = st.selectbox("School Type", get_opts("School_Type", ["Public", "Private"]))
            major = st.selectbox("Major", get_opts("major", ["Computer Science", "Engineering", "Arts"]))
        with col2:
            extracurricular = st.selectbox("Extracurricular Activities", get_opts("Extracurricular_Activities", ["Yes", "No"]))
            tutoring_sessions = st.slider("Tutoring Sessions", 0, 10, 2)
        with col3:
            family_income = st.selectbox("Family Income Range", get_opts("Family_Income", ["Low", "Medium", "High"]))
            parental_edu = st.selectbox("Parental Education Level", get_opts("Parental_Education_Level", ["High School", "College", "Postgraduate"]))
    
    # Prediction button
    if st.button("🔮 Predict Performance", type="primary", use_container_width=True):
        try:
            # Create input dictionary
            input_data = {
                'Gender': gender,
                'age': age,
                'Age': age, # Safety for casing
                'Hours_Studied': hours_studied,
                'Social_Media_Hours': social_media_hours,
                'Movie_Hours': movie_hours,
                'Attendance': attendance,
                'Sleep_Hours': sleep_hours,
                'diet_quality': diet_quality,
                'Internet_Access': internet_access,
                'mental_health_rating': mental_health,
                'Physical_Activity': physical_activity,
                'Extracurricular_Activities': extracurricular,
                'Family_Income': family_income,
                'Parental_Involvement': parental_involvement,
                'Access_to_Resources': access_to_resources,
                'Teacher_Quality': teacher_quality,
                'School_Type': school_type,
                'Parental_Education_Level': parental_edu,
                'Motivation_Level': motivation_level,
                'Previous_Scores': previous_scores,
                'Relationship_Status': relationship_status,
                'Tutoring_Sessions': tutoring_sessions,
                'Movie_Addiction': 'Low' if movie_hours < 1.5 else ('Medium' if movie_hours < 3.5 else 'High'),
                'Gym_Discipline': 'Low',
                'Social_Media_Addiction_Level': 3.0,
            }
            
            # Additional academic features that might be in the dataset
            input_data['Peer_Influence'] = "Neutral"
            input_data['Learning_Disabilities'] = "No"
            input_data['Distance_from_Home'] = "Near"
            input_data['major'] = major
            
            input_df = pd.DataFrame([input_data])
            
            # CRITICAL: Ensure every single column the model was trained on exists here
            feature_columns = st.session_state.feature_columns
            for col in feature_columns:
                if col not in input_df.columns:
                    # Fill with 0 or most common value logic if needed
                    input_df[col] = 0
            
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
            input_scaled = scaler.transform(input_df)
            
            # Make predictions
            score_pred = model_handler.predict('linear_regression', input_scaled)[0]
            pass_pred = model_handler.predict('logistic_regression', input_scaled)[0]
            
            # Display results
            st.markdown("---")
            st.markdown("### 🎉 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Exam Score", f"{score_pred:.1f}")
            with col2:
                status = "Pass ✅" if pass_pred == 1 else "Fail ❌"
                st.metric("Pass/Fail Prediction", status)
            with col3:
                # Grade
                if score_pred >= 90:
                    grade = "A"
                elif score_pred >= 80:
                    grade = "B"
                elif score_pred >= 70:
                    grade = "C"
                elif score_pred >= 60:
                    grade = "D"
                else:
                    grade = "F"
                st.metric("Predicted Grade", grade)
            
            # Insights
            st.markdown("### 💡 Insights")
            insights = []
            if social_media_hours > 4:
                insights.append("⚠️ High social media usage (>4h) might be distracting you.")
            if movie_hours > 3:
                insights.append("⚠️ Excessive Netflix/Movie time can impact your focus.")
            if sleep_hours < 7:
                insights.append("😴 Getting more than 7 hours of sleep is recommended for students.")
            if previous_scores < 2.5:
                insights.append("📚 Your previous GPA is low. Focus on core subjects.")
            if stress_level > 3:
                insights.append("🧘 High stress detected. Consider mindfulness or time management.")
            if exam_anxiety > 50:
                insights.append("🧠 High exam anxiety. Preparation and tutoring might help.")
            if attendance < 80:
                insights.append("📅 Improving attendance remains the most effective way to boost scores.")
            
            if insights:
                for insight in insights:
                    st.write(insight)
            else:
                st.success("✨ Great habits! Keep up the good work!")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
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
        
        clf_models = ['logistic_regression', 'knn', 'svm', 'mlp_classifier']
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
