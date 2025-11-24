"""
Unit tests for the course recommendation system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from course_recommender.data import DataLoader, DataProcessor
from course_recommender.models import ContentBasedRecommender, CollaborativeRecommender, HybridRecommender
from course_recommender.evaluation import Metrics, Evaluator
from course_recommender.utils import set_seed, load_config, save_config


class TestDataLoader:
    """Test cases for DataLoader class."""
    
    def test_init(self):
        """Test DataLoader initialization."""
        loader = DataLoader("test_data")
        assert loader.data_dir == "test_data"
        assert loader.interactions_df is None
        assert loader.items_df is None
        assert loader.users_df is None
        
    @patch('pandas.read_csv')
    def test_load_interactions(self, mock_read_csv):
        """Test loading interactions data."""
        mock_df = pd.DataFrame({
            'user_id': ['user1', 'user2'],
            'item_id': ['item1', 'item2'],
            'rating': [4, 5]
        })
        mock_read_csv.return_value = mock_df
        
        loader = DataLoader("test_data")
        result = loader.load_interactions("interactions.csv")
        
        assert result.equals(mock_df)
        assert loader.interactions_df.equals(mock_df)
        mock_read_csv.assert_called_once_with("interactions.csv")
        
    @patch('pandas.read_csv')
    def test_load_items(self, mock_read_csv):
        """Test loading items data."""
        mock_df = pd.DataFrame({
            'item_id': ['item1', 'item2'],
            'title': ['Course 1', 'Course 2'],
            'description': ['Desc 1', 'Desc 2']
        })
        mock_read_csv.return_value = mock_df
        
        loader = DataLoader("test_data")
        result = loader.load_items("items.csv")
        
        assert result.equals(mock_df)
        assert loader.items_df.equals(mock_df)
        mock_read_csv.assert_called_once_with("items.csv")


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    def test_init(self):
        """Test DataProcessor initialization."""
        processor = DataProcessor()
        assert processor.user_encoder is not None
        assert processor.item_encoder is not None
        assert processor.is_fitted is False
        
    def test_fit_transform_interactions(self):
        """Test fitting and transforming interactions."""
        interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user2', 'user1'],
            'item_id': ['item1', 'item2', 'item3'],
            'rating': [4, 5, 3]
        })
        
        processor = DataProcessor()
        result = processor.fit_transform_interactions(interactions_df)
        
        assert 'user_id_encoded' in result.columns
        assert 'item_id_encoded' in result.columns
        assert processor.is_fitted is True
        assert len(result) == 3
        
    def test_create_train_test_split(self):
        """Test creating train-test split."""
        interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user2', 'user3', 'user4'],
            'item_id': ['item1', 'item2', 'item3', 'item4'],
            'rating': [4, 5, 3, 4],
            'user_id_encoded': [0, 1, 2, 3],
            'item_id_encoded': [0, 1, 2, 3]
        })
        
        processor = DataProcessor()
        train_df, test_df = processor.create_train_test_split(interactions_df, test_size=0.5)
        
        assert len(train_df) + len(test_df) == len(interactions_df)
        assert len(train_df) >= 1
        assert len(test_df) >= 1


class TestContentBasedRecommender:
    """Test cases for ContentBasedRecommender class."""
    
    def test_init(self):
        """Test ContentBasedRecommender initialization."""
        model = ContentBasedRecommender(use_sentence_transformers=False)
        assert model.use_sentence_transformers is False
        assert model.max_features == 1000
        assert model.is_fitted is False
        
    def test_fit(self):
        """Test fitting the content-based model."""
        items_df = pd.DataFrame({
            'item_id': ['item1', 'item2'],
            'title': ['Python Course', 'Data Science Course'],
            'description': ['Learn Python programming', 'Learn data science'],
            'category': ['Programming', 'Data Science'],
            'difficulty': [2, 3],
            'duration_hours': [20, 40],
            'price': [0, 99],
            'rating': [4.5, 4.2],
            'tags': ['python|beginner', 'data|intermediate']
        })
        
        model = ContentBasedRecommender(use_sentence_transformers=False)
        model.fit(items_df)
        
        assert model.is_fitted is True
        assert model.item_features is not None
        assert model.item_ids == ['item1', 'item2']
        
    def test_recommend(self):
        """Test generating recommendations."""
        items_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3'],
            'title': ['Python Course', 'Data Science Course', 'ML Course'],
            'description': ['Learn Python', 'Learn data science', 'Learn ML'],
            'category': ['Programming', 'Data Science', 'ML'],
            'difficulty': [2, 3, 4],
            'duration_hours': [20, 40, 60],
            'price': [0, 99, 199],
            'rating': [4.5, 4.2, 4.8],
            'tags': ['python|beginner', 'data|intermediate', 'ml|advanced']
        })
        
        user_interactions = pd.DataFrame({
            'user_id': ['user1'],
            'item_id': ['item1'],
            'rating': [5]
        })
        
        model = ContentBasedRecommender(use_sentence_transformers=False)
        model.fit(items_df)
        
        recommendations = model.recommend(
            user_id="user1",
            user_interactions=user_interactions,
            items_df=items_df,
            n_recommendations=2
        )
        
        assert len(recommendations) <= 2
        assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)


class TestCollaborativeRecommender:
    """Test cases for CollaborativeRecommender class."""
    
    def test_init(self):
        """Test CollaborativeRecommender initialization."""
        model = CollaborativeRecommender()
        assert model.n_factors == 50
        assert model.n_iterations == 20
        assert model.is_fitted is False
        
    def test_fit(self):
        """Test fitting the collaborative model."""
        interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user2', 'user1', 'user2'],
            'item_id': ['item1', 'item1', 'item2', 'item2'],
            'rating': [4, 5, 3, 4]
        })
        
        model = CollaborativeRecommender(n_iterations=1)  # Reduce iterations for testing
        model.fit(interactions_df)
        
        assert model.is_fitted is True
        assert model.user_factors is not None
        assert model.item_factors is not None
        
    def test_recommend(self):
        """Test generating recommendations."""
        interactions_df = pd.DataFrame({
            'user_id': ['user1', 'user2', 'user1', 'user2'],
            'item_id': ['item1', 'item1', 'item2', 'item2'],
            'rating': [4, 5, 3, 4]
        })
        
        model = CollaborativeRecommender(n_iterations=1)
        model.fit(interactions_df)
        
        recommendations = model.recommend(
            user_id="user1",
            n_recommendations=2,
            exclude_interacted=True,
            interactions_df=interactions_df
        )
        
        assert len(recommendations) <= 2
        assert all(isinstance(rec, tuple) and len(rec) == 2 for rec in recommendations)


class TestMetrics:
    """Test cases for Metrics class."""
    
    def test_precision_at_k(self):
        """Test precision@k calculation."""
        recommended_items = ['item1', 'item2', 'item3']
        relevant_items = ['item1', 'item3', 'item4']
        
        precision = Metrics.precision_at_k(recommended_items, relevant_items, k=3)
        assert precision == 2/3  # 2 relevant items out of 3 recommendations
        
    def test_recall_at_k(self):
        """Test recall@k calculation."""
        recommended_items = ['item1', 'item2', 'item3']
        relevant_items = ['item1', 'item3', 'item4']
        
        recall = Metrics.recall_at_k(recommended_items, relevant_items, k=3)
        assert recall == 2/3  # 2 relevant items out of 3 total relevant items
        
    def test_ndcg_at_k(self):
        """Test NDCG@k calculation."""
        recommended_items = ['item1', 'item2', 'item3']
        relevant_items = ['item1', 'item3', 'item4']
        
        ndcg = Metrics.ndcg_at_k(recommended_items, relevant_items, k=3)
        assert 0 <= ndcg <= 1
        
    def test_hit_rate_at_k(self):
        """Test hit rate@k calculation."""
        recommended_items = ['item1', 'item2', 'item3']
        relevant_items = ['item1', 'item3', 'item4']
        
        hit_rate = Metrics.hit_rate_at_k(recommended_items, relevant_items, k=3)
        assert hit_rate == 1.0  # At least one relevant item in recommendations
        
    def test_coverage(self):
        """Test coverage calculation."""
        recommended_items_all_users = [
            ['item1', 'item2'],
            ['item2', 'item3'],
            ['item1', 'item3']
        ]
        all_items = ['item1', 'item2', 'item3', 'item4']
        
        coverage = Metrics.coverage(recommended_items_all_users, all_items)
        assert coverage == 3/4  # 3 unique items recommended out of 4 total items


class TestEvaluator:
    """Test cases for Evaluator class."""
    
    def test_init(self):
        """Test Evaluator initialization."""
        evaluator = Evaluator(k_values=[5, 10])
        assert evaluator.k_values == [5, 10]
        assert evaluator.metrics is not None
        
    def test_evaluate_model(self):
        """Test model evaluation."""
        # Create mock model
        mock_model = Mock()
        mock_model.recommend.return_value = [('item1', 0.8), ('item2', 0.7)]
        
        # Create test data
        test_interactions = pd.DataFrame({
            'user_id': ['user1', 'user1'],
            'item_id': ['item1', 'item2'],
            'rating': [4, 5]
        })
        
        items_df = pd.DataFrame({
            'item_id': ['item1', 'item2', 'item3'],
            'title': ['Course 1', 'Course 2', 'Course 3'],
            'description': ['Desc 1', 'Desc 2', 'Desc 3'],
            'category': ['Cat 1', 'Cat 2', 'Cat 3'],
            'difficulty': [2, 3, 4],
            'duration_hours': [20, 40, 60],
            'price': [0, 99, 199],
            'rating': [4.5, 4.2, 4.8],
            'tags': ['tag1', 'tag2', 'tag3']
        })
        
        evaluator = Evaluator(k_values=[5])
        results = evaluator.evaluate_model(
            mock_model, test_interactions, items_df, model_name="test_model"
        )
        
        assert results["model_name"] == "test_model"
        assert "metrics" in results
        assert "precision@5" in results["metrics"]


class TestUtils:
    """Test cases for utility functions."""
    
    def test_set_seed(self):
        """Test setting random seed."""
        set_seed(42)
        # This is hard to test directly, but we can ensure it doesn't raise an error
        assert True
        
    def test_load_config(self):
        """Test loading configuration."""
        # Create a temporary config file
        import tempfile
        import yaml
        
        config_data = {"test": "value", "nested": {"key": "value"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
            
        try:
            config = load_config(config_path)
            assert config == config_data
        finally:
            os.unlink(config_path)
            
    def test_save_config(self):
        """Test saving configuration."""
        import tempfile
        
        config_data = {"test": "value", "nested": {"key": "value"}}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
            
        try:
            save_config(config_data, config_path)
            
            # Load it back to verify
            loaded_config = load_config(config_path)
            assert loaded_config == config_data
        finally:
            os.unlink(config_path)


if __name__ == "__main__":
    pytest.main([__file__])
