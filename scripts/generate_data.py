"""
Script to generate realistic course recommendation dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_courses_data(n_courses: int = 100) -> pd.DataFrame:
    """Generate realistic course data.
    
    Args:
        n_courses: Number of courses to generate.
        
    Returns:
        DataFrame containing course information.
    """
    # Course categories and their characteristics
    categories = {
        "Programming": {
            "difficulty": [1, 2, 3, 4, 5],
            "duration_hours": [10, 20, 40, 60, 80],
            "prerequisites": ["None", "Basic Computer Skills", "Programming Basics", "Intermediate Programming"]
        },
        "Data Science": {
            "difficulty": [2, 3, 4, 5],
            "duration_hours": [20, 40, 60, 80],
            "prerequisites": ["Statistics", "Python Basics", "Mathematics", "Machine Learning Basics"]
        },
        "Machine Learning": {
            "difficulty": [3, 4, 5],
            "duration_hours": [40, 60, 80],
            "prerequisites": ["Linear Algebra", "Statistics", "Python", "Data Science"]
        },
        "Web Development": {
            "difficulty": [1, 2, 3, 4],
            "duration_hours": [15, 30, 45, 60],
            "prerequisites": ["HTML/CSS", "JavaScript", "None", "Basic Programming"]
        },
        "Design": {
            "difficulty": [1, 2, 3],
            "duration_hours": [10, 20, 30],
            "prerequisites": ["None", "Design Basics", "Creative Thinking"]
        }
    }
    
    courses = []
    course_id = 1
    
    for category, props in categories.items():
        n_category_courses = n_courses // len(categories)
        
        for i in range(n_category_courses):
            difficulty = random.choice(props["difficulty"])
            duration = random.choice(props["duration_hours"])
            prerequisite = random.choice(props["prerequisites"])
            
            # Generate course title
            if category == "Programming":
                titles = [
                    f"Python Programming Fundamentals",
                    f"Advanced Python Development",
                    f"JavaScript Mastery",
                    f"Java for Beginners",
                    f"C++ Programming",
                    f"React Development",
                    f"Node.js Backend Development"
                ]
            elif category == "Data Science":
                titles = [
                    f"Data Analysis with Pandas",
                    f"Statistical Modeling",
                    f"Data Visualization",
                    f"Big Data Processing",
                    f"Time Series Analysis",
                    f"Experimental Design"
                ]
            elif category == "Machine Learning":
                titles = [
                    f"Introduction to Machine Learning",
                    f"Deep Learning with TensorFlow",
                    f"Natural Language Processing",
                    f"Computer Vision",
                    f"Reinforcement Learning",
                    f"MLOps and Deployment"
                ]
            elif category == "Web Development":
                titles = [
                    f"HTML/CSS Fundamentals",
                    f"JavaScript ES6+",
                    f"React Frontend Development",
                    f"Django Web Framework",
                    f"Full-Stack Development",
                    f"Web Performance Optimization"
                ]
            else:  # Design
                titles = [
                    f"UI/UX Design Principles",
                    f"Graphic Design Basics",
                    f"User Research Methods",
                    f"Prototyping with Figma",
                    f"Design Systems"
                ]
            
            title = random.choice(titles)
            
            # Generate description
            descriptions = {
                "Programming": f"Learn {title.lower()} with hands-on projects and real-world applications. Perfect for {prerequisite.lower()}.",
                "Data Science": f"Master {title.lower()} using industry-standard tools and techniques. Build your data science portfolio.",
                "Machine Learning": f"Comprehensive course on {title.lower()}. From theory to implementation with practical projects.",
                "Web Development": f"Build modern web applications with {title.lower()}. Create responsive and interactive websites.",
                "Design": f"Learn {title.lower()} and create beautiful, user-friendly designs. Develop your creative skills."
            }
            
            description = descriptions[category]
            
            # Generate tags
            tags = [category.lower()]
            if difficulty <= 2:
                tags.append("beginner")
            elif difficulty <= 4:
                tags.append("intermediate")
            else:
                tags.append("advanced")
                
            if duration <= 20:
                tags.append("short")
            elif duration <= 40:
                tags.append("medium")
            else:
                tags.append("long")
                
            courses.append({
                "item_id": f"course_{course_id:03d}",
                "title": title,
                "description": description,
                "category": category,
                "difficulty": difficulty,
                "duration_hours": duration,
                "prerequisites": prerequisite,
                "tags": "|".join(tags),
                "price": random.choice([0, 29, 49, 99, 199, 299]),
                "rating": round(random.uniform(3.5, 5.0), 1),
                "enrollment_count": random.randint(100, 10000)
            })
            
            course_id += 1
    
    return pd.DataFrame(courses)


def generate_users_data(n_users: int = 1000) -> pd.DataFrame:
    """Generate realistic user data.
    
    Args:
        n_users: Number of users to generate.
        
    Returns:
        DataFrame containing user information.
    """
    users = []
    
    # User profiles with different learning preferences
    profiles = [
        {"experience_level": "beginner", "interests": ["programming", "web development"], "learning_style": "hands-on"},
        {"experience_level": "intermediate", "interests": ["data science", "machine learning"], "learning_style": "theoretical"},
        {"experience_level": "advanced", "interests": ["machine learning", "programming"], "learning_style": "project-based"},
        {"experience_level": "beginner", "interests": ["design", "web development"], "learning_style": "visual"},
        {"experience_level": "intermediate", "interests": ["programming", "data science"], "learning_style": "practical"}
    ]
    
    for i in range(n_users):
        profile = random.choice(profiles)
        
        users.append({
            "user_id": f"user_{i+1:04d}",
            "age": random.randint(18, 65),
            "experience_level": profile["experience_level"],
            "primary_interest": random.choice(profile["interests"]),
            "learning_style": profile["learning_style"],
            "preferred_difficulty": random.choice([1, 2, 3, 4, 5]),
            "max_course_duration": random.choice([20, 40, 60, 80]),
            "budget_preference": random.choice(["free", "low", "medium", "high"])
        })
    
    return pd.DataFrame(users)


def generate_interactions_data(
    users_df: pd.DataFrame, 
    courses_df: pd.DataFrame, 
    n_interactions: int = 5000
) -> pd.DataFrame:
    """Generate realistic user-course interactions.
    
    Args:
        users_df: DataFrame containing user information.
        courses_df: DataFrame containing course information.
        n_interactions: Number of interactions to generate.
        
    Returns:
        DataFrame containing user-course interactions.
    """
    interactions = []
    
    # Generate interactions based on user preferences
    for _ in range(n_interactions):
        user = users_df.sample(1).iloc[0]
        
        # Filter courses based on user preferences
        suitable_courses = courses_df.copy()
        
        # Filter by difficulty preference
        if user["preferred_difficulty"] <= 2:
            suitable_courses = suitable_courses[suitable_courses["difficulty"] <= 3]
        elif user["preferred_difficulty"] >= 4:
            suitable_courses = suitable_courses[suitable_courses["difficulty"] >= 3]
            
        # Filter by duration preference
        suitable_courses = suitable_courses[
            suitable_courses["duration_hours"] <= user["max_course_duration"]
        ]
        
        # Filter by budget preference
        if user["budget_preference"] == "free":
            suitable_courses = suitable_courses[suitable_courses["price"] == 0]
        elif user["budget_preference"] == "low":
            suitable_courses = suitable_courses[suitable_courses["price"] <= 49]
        elif user["budget_preference"] == "medium":
            suitable_courses = suitable_courses[suitable_courses["price"] <= 199]
        
        # Filter by interest
        if user["primary_interest"] in ["programming", "web development"]:
            suitable_courses = suitable_courses[
                suitable_courses["category"].isin(["Programming", "Web Development"])
            ]
        elif user["primary_interest"] in ["data science", "machine learning"]:
            suitable_courses = suitable_courses[
                suitable_courses["category"].isin(["Data Science", "Machine Learning"])
            ]
        elif user["primary_interest"] == "design":
            suitable_courses = suitable_courses[
                suitable_courses["category"] == "Design"
            ]
        
        if len(suitable_courses) == 0:
            # If no suitable courses, pick any course
            course = courses_df.sample(1).iloc[0]
        else:
            course = suitable_courses.sample(1).iloc[0]
        
        # Generate rating based on course quality and user preferences
        base_rating = course["rating"]
        
        # Adjust rating based on difficulty match
        difficulty_diff = abs(user["preferred_difficulty"] - course["difficulty"])
        difficulty_bonus = max(0, 0.5 - difficulty_diff * 0.1)
        
        # Adjust rating based on interest match
        interest_bonus = 0.2 if user["primary_interest"] in course["tags"] else 0
        
        final_rating = min(5.0, base_rating + difficulty_bonus + interest_bonus + random.uniform(-0.3, 0.3))
        final_rating = max(1.0, final_rating)
        
        # Generate timestamp (last 2 years)
        start_date = datetime.now() - timedelta(days=730)
        random_days = random.randint(0, 730)
        timestamp = start_date + timedelta(days=random_days)
        
        interactions.append({
            "user_id": user["user_id"],
            "item_id": course["item_id"],
            "rating": round(final_rating, 1),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "interaction_type": random.choice(["enrollment", "completion", "review", "bookmark"]),
            "duration_minutes": random.randint(5, 180) if random.random() > 0.3 else None
        })
    
    return pd.DataFrame(interactions)


def main():
    """Generate and save the complete dataset."""
    logger.info("Generating course recommendation dataset...")
    
    # Generate data
    courses_df = generate_courses_data(n_courses=100)
    users_df = generate_users_data(n_users=1000)
    interactions_df = generate_interactions_data(users_df, courses_df, n_interactions=5000)
    
    # Save to CSV files
    courses_df.to_csv("data/raw/items.csv", index=False)
    users_df.to_csv("data/raw/users.csv", index=False)
    interactions_df.to_csv("data/raw/interactions.csv", index=False)
    
    logger.info(f"Generated dataset:")
    logger.info(f"  - {len(courses_df)} courses")
    logger.info(f"  - {len(users_df)} users")
    logger.info(f"  - {len(interactions_df)} interactions")
    logger.info("Dataset saved to data/raw/ directory")


if __name__ == "__main__":
    main()
