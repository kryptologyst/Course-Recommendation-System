"""
Streamlit demo application for the course recommendation system.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from course_recommender.data import DataLoader, DataProcessor
from course_recommender.models import ContentBasedRecommender, CollaborativeRecommender, HybridRecommender
from course_recommender.evaluation import Evaluator
from course_recommender.utils import set_seed

# Set page config
st.set_page_config(
    page_title="Course Recommendation System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set random seed
set_seed(42)


@st.cache_data
def load_data():
    """Load and cache the dataset."""
    try:
        loader = DataLoader("data/raw")
        interactions_df = loader.load_interactions("interactions.csv")
        items_df = loader.load_items("items.csv")
        users_df = loader.load_users("users.csv")
        return interactions_df, items_df, users_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


@st.cache_resource
def load_models(interactions_df, items_df, users_df):
    """Load and cache the recommendation models."""
    try:
        # Process data
        processor = DataProcessor()
        processed_interactions = processor.fit_transform_interactions(interactions_df)
        
        # Split data
        train_df, test_df = processor.create_train_test_split(processed_interactions)
        
        # Initialize models
        content_model = ContentBasedRecommender(use_sentence_transformers=False)  # Use TF-IDF for demo
        collaborative_model = CollaborativeRecommender()
        hybrid_model = HybridRecommender()
        
        # Fit models
        content_model.fit(items_df)
        collaborative_model.fit(train_df)
        hybrid_model.fit(train_df, items_df, users_df)
        
        return {
            "content": content_model,
            "collaborative": collaborative_model,
            "hybrid": hybrid_model
        }, processor, train_df, test_df
        
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None


def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("🎓 Course Recommendation System")
    st.markdown("""
    A comprehensive course recommendation system that uses multiple approaches:
    - **Content-based filtering**: Recommends courses based on course descriptions and user preferences
    - **Collaborative filtering**: Recommends courses based on similar users' preferences
    - **Hybrid approach**: Combines both methods for better recommendations
    """)
    
    # Load data
    with st.spinner("Loading data..."):
        interactions_df, items_df, users_df = load_data()
        
    if interactions_df is None:
        st.error("Failed to load data. Please ensure the data files exist in data/raw/")
        return
        
    # Load models
    with st.spinner("Loading models..."):
        models, processor, train_df, test_df = load_models(interactions_df, items_df, users_df)
        
    if models is None:
        st.error("Failed to load models.")
        return
        
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["Recommendations", "Model Comparison", "Data Analysis", "About"]
    )
    
    if page == "Recommendations":
        show_recommendations_page(models, items_df, users_df, interactions_df, processor)
    elif page == "Model Comparison":
        show_model_comparison_page(models, test_df, items_df, users_df)
    elif page == "Data Analysis":
        show_data_analysis_page(interactions_df, items_df, users_df)
    elif page == "About":
        show_about_page()


def show_recommendations_page(models, items_df, users_df, interactions_df, processor):
    """Show the recommendations page."""
    st.header("📚 Get Course Recommendations")
    
    # User selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_user = st.selectbox(
            "Select a user:",
            options=users_df["user_id"].tolist(),
            format_func=lambda x: f"{x} - {users_df[users_df['user_id'] == x]['primary_interest'].iloc[0]}"
        )
        
    with col2:
        n_recommendations = st.slider("Number of recommendations:", 5, 20, 10)
        
    # Model selection
    model_type = st.selectbox(
        "Select recommendation model:",
        ["content", "collaborative", "hybrid"],
        format_func=lambda x: {
            "content": "Content-based Filtering",
            "collaborative": "Collaborative Filtering", 
            "hybrid": "Hybrid Approach"
        }[x]
    )
    
    # Get user information
    user_info = users_df[users_df["user_id"] == selected_user].iloc[0]
    
    st.subheader(f"User Profile: {selected_user}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Experience Level", user_info["experience_level"].title())
    with col2:
        st.metric("Primary Interest", user_info["primary_interest"].title())
    with col3:
        st.metric("Learning Style", user_info["learning_style"].title())
    with col4:
        st.metric("Preferred Difficulty", f"{user_info['preferred_difficulty']}/5")
        
    # Get user's interaction history
    user_interactions = interactions_df[interactions_df["user_id"] == selected_user]
    
    if len(user_interactions) > 0:
        st.subheader("Previous Course Interactions")
        
        # Show user's previous courses
        user_courses = user_interactions.merge(items_df, on="item_id")[["title", "category", "rating", "interaction_type"]]
        st.dataframe(user_courses, use_container_width=True)
        
    # Generate recommendations
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            try:
                model = models[model_type]
                
                if model_type == "content":
                    recommendations = model.recommend(
                        selected_user, user_interactions, items_df, n_recommendations=n_recommendations
                    )
                elif model_type == "collaborative":
                    recommendations = model.recommend(
                        selected_user, n_recommendations=n_recommendations, 
                        exclude_interacted=True, interactions_df=interactions_df
                    )
                else:  # hybrid
                    recommendations = model.recommend(
                        selected_user, interactions_df, items_df, users_df, n_recommendations=n_recommendations
                    )
                    
                # Display recommendations
                st.subheader(f"Top {n_recommendations} Recommendations")
                
                for i, (item_id, score) in enumerate(recommendations, 1):
                    item_info = items_df[items_df["item_id"] == item_id].iloc[0]
                    
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.write(f"**{i}. {item_info['title']}**")
                            st.write(f"Category: {item_info['category']}")
                            st.write(f"Difficulty: {item_info['difficulty']}/5")
                            st.write(f"Duration: {item_info['duration_hours']} hours")
                            st.write(f"Rating: {item_info['rating']}/5")
                            st.write(f"Description: {item_info['description']}")
                            
                        with col2:
                            st.metric("Recommendation Score", f"{score:.3f}")
                            
                        st.divider()
                        
            except Exception as e:
                st.error(f"Error generating recommendations: {e}")


def show_model_comparison_page(models, test_df, items_df, users_df):
    """Show the model comparison page."""
    st.header("📊 Model Performance Comparison")
    
    st.markdown("""
    This page compares the performance of different recommendation models using various metrics:
    - **Precision@K**: Fraction of recommended items that are relevant
    - **Recall@K**: Fraction of relevant items that are recommended
    - **NDCG@K**: Normalized Discounted Cumulative Gain
    - **MAP@K**: Mean Average Precision
    - **Hit Rate@K**: Fraction of users for whom at least one relevant item is recommended
    - **Coverage**: Fraction of catalog items that are recommended
    """)
    
    if st.button("Run Model Evaluation", type="primary"):
        with st.spinner("Evaluating models..."):
            try:
                evaluator = Evaluator(k_values=[5, 10, 20])
                
                # Evaluate each model
                results = {}
                for model_name, model in models.items():
                    results[model_name] = evaluator.evaluate_model(
                        model, test_df, items_df, users_df, n_recommendations=20, model_name=model_name
                    )
                    
                # Create comparison DataFrame
                comparison_data = []
                for result in results.values():
                    row = {"Model": result["model_name"].title()}
                    row.update(result["metrics"])
                    comparison_data.append(row)
                    
                comparison_df = pd.DataFrame(comparison_data)
                
                # Display results
                st.subheader("Performance Metrics")
                st.dataframe(comparison_df, use_container_width=True)
                
                # Create visualizations
                st.subheader("Performance Visualization")
                
                # Select metrics to plot
                metric_cols = [col for col in comparison_df.columns if col != "Model"]
                selected_metrics = st.multiselect(
                    "Select metrics to visualize:",
                    metric_cols,
                    default=["precision@10", "recall@10", "ndcg@10"]
                )
                
                if selected_metrics:
                    # Create bar chart
                    fig = go.Figure()
                    
                    for metric in selected_metrics:
                        fig.add_trace(go.Bar(
                            name=metric,
                            x=comparison_df["Model"],
                            y=comparison_df[metric],
                            text=comparison_df[metric].round(4),
                            textposition='auto'
                        ))
                        
                    fig.update_layout(
                        title="Model Performance Comparison",
                        xaxis_title="Model",
                        yaxis_title="Score",
                        barmode='group',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                # Generate report
                st.subheader("Evaluation Report")
                report = evaluator.generate_report(comparison_df)
                st.text(report)
                
            except Exception as e:
                st.error(f"Error during evaluation: {e}")


def show_data_analysis_page(interactions_df, items_df, users_df):
    """Show the data analysis page."""
    st.header("📈 Data Analysis")
    
    # Overview statistics
    st.subheader("Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Users", len(users_df))
    with col2:
        st.metric("Total Courses", len(items_df))
    with col3:
        st.metric("Total Interactions", len(interactions_df))
    with col4:
        st.metric("Avg Rating", f"{interactions_df['rating'].mean():.2f}")
        
    # Course distribution by category
    st.subheader("Course Distribution by Category")
    category_counts = items_df["category"].value_counts()
    
    fig = px.pie(
        values=category_counts.values,
        names=category_counts.index,
        title="Courses by Category"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Difficulty distribution
    st.subheader("Course Difficulty Distribution")
    difficulty_counts = items_df["difficulty"].value_counts().sort_index()
    
    fig = px.bar(
        x=difficulty_counts.index,
        y=difficulty_counts.values,
        title="Number of Courses by Difficulty Level",
        labels={"x": "Difficulty Level", "y": "Number of Courses"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # User experience level distribution
    st.subheader("User Experience Level Distribution")
    exp_counts = users_df["experience_level"].value_counts()
    
    fig = px.bar(
        x=exp_counts.index,
        y=exp_counts.values,
        title="Number of Users by Experience Level",
        labels={"x": "Experience Level", "y": "Number of Users"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Rating distribution
    st.subheader("Rating Distribution")
    rating_counts = interactions_df["rating"].value_counts().sort_index()
    
    fig = px.histogram(
        interactions_df,
        x="rating",
        title="Distribution of Ratings",
        labels={"x": "Rating", "y": "Count"}
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top rated courses
    st.subheader("Top Rated Courses")
    top_courses = interactions_df.groupby("item_id")["rating"].agg(["mean", "count"]).reset_index()
    top_courses = top_courses.merge(items_df[["item_id", "title", "category"]], on="item_id")
    top_courses = top_courses[top_courses["count"] >= 5].sort_values("mean", ascending=False).head(10)
    
    st.dataframe(top_courses, use_container_width=True)


def show_about_page():
    """Show the about page."""
    st.header("About This System")
    
    st.markdown("""
    ## Course Recommendation System
    
    This is a comprehensive course recommendation system that demonstrates different approaches to recommendation:
    
    ### Models Implemented
    
    1. **Content-based Filtering**
       - Uses TF-IDF vectorization of course descriptions
       - Creates user profiles based on interaction history
       - Recommends courses similar to user's preferences
       
    2. **Collaborative Filtering**
       - Uses matrix factorization to find latent user-item relationships
       - Recommends courses liked by similar users
       - Handles cold-start problems with bias terms
       
    3. **Hybrid Approach**
       - Combines content-based and collaborative filtering
       - Uses weighted averaging or machine learning blending
       - Provides explanations for recommendations
       
    ### Features
    
    - **Multiple Recommendation Approaches**: Compare different algorithms
    - **Comprehensive Evaluation**: Multiple metrics including Precision@K, Recall@K, NDCG@K, MAP@K
    - **Interactive Demo**: Streamlit-based user interface
    - **Data Analysis**: Visualizations and insights about the dataset
    - **Model Comparison**: Side-by-side performance comparison
    
    ### Technical Stack
    
    - **Python 3.10+**: Modern Python with type hints
    - **scikit-learn**: Machine learning algorithms and utilities
    - **pandas**: Data manipulation and analysis
    - **numpy**: Numerical computing
    - **streamlit**: Interactive web application
    - **plotly**: Interactive visualizations
    - **sentence-transformers**: Advanced text embeddings (optional)
    
    ### Dataset
    
    The system uses a synthetic dataset with:
    - 1,000 users with different experience levels and preferences
    - 100 courses across different categories (Programming, Data Science, Machine Learning, Web Development, Design)
    - 5,000 interactions with ratings and timestamps
    
    ### Usage
    
    1. **Get Recommendations**: Select a user and model to get personalized course recommendations
    2. **Compare Models**: Evaluate and compare different recommendation approaches
    3. **Analyze Data**: Explore the dataset with interactive visualizations
    
    ### Future Enhancements
    
    - Real-time recommendation updates
    - Advanced deep learning models
    - Multi-criteria recommendations
    - Explainable AI features
    - A/B testing framework
    """)


if __name__ == "__main__":
    main()
