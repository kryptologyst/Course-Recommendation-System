"""
Course Recommendation System

A modern course recommendation system that implements multiple recommendation approaches:
- Content-based filtering using TF-IDF and sentence transformers
- Collaborative filtering using matrix factorization
- Hybrid approaches combining both methods

This package provides a complete pipeline for course recommendation including
data processing, model training, evaluation, and interactive demos.
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .data import DataLoader, DataProcessor
from .models import (
    ContentBasedRecommender,
    CollaborativeRecommender,
    HybridRecommender,
)
from .evaluation import Evaluator, Metrics
from .utils import set_seed, load_config

__all__ = [
    "DataLoader",
    "DataProcessor", 
    "ContentBasedRecommender",
    "CollaborativeRecommender",
    "HybridRecommender",
    "Evaluator",
    "Metrics",
    "set_seed",
    "load_config",
]
