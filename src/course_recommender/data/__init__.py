"""
Data loading and processing utilities for the course recommendation system.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading and basic preprocessing of course recommendation data."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the data loader.
        
        Args:
            data_dir: Directory containing the data files.
        """
        self.data_dir = data_dir
        self.interactions_df: Optional[pd.DataFrame] = None
        self.items_df: Optional[pd.DataFrame] = None
        self.users_df: Optional[pd.DataFrame] = None
        
    def load_interactions(self, file_path: str) -> pd.DataFrame:
        """Load user-item interactions data.
        
        Args:
            file_path: Path to the interactions CSV file.
            
        Returns:
            DataFrame containing user-item interactions.
        """
        try:
            self.interactions_df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.interactions_df)} interactions")
            return self.interactions_df
        except FileNotFoundError:
            logger.error(f"Interactions file not found: {file_path}")
            raise
            
    def load_items(self, file_path: str) -> pd.DataFrame:
        """Load course items data.
        
        Args:
            file_path: Path to the items CSV file.
            
        Returns:
            DataFrame containing course information.
        """
        try:
            self.items_df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.items_df)} items")
            return self.items_df
        except FileNotFoundError:
            logger.error(f"Items file not found: {file_path}")
            raise
            
    def load_users(self, file_path: str) -> pd.DataFrame:
        """Load user data.
        
        Args:
            file_path: Path to the users CSV file.
            
        Returns:
            DataFrame containing user information.
        """
        try:
            self.users_df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.users_df)} users")
            return self.users_df
        except FileNotFoundError:
            logger.error(f"Users file not found: {file_path}")
            raise


class DataProcessor:
    """Handles data preprocessing and feature engineering."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.is_fitted = False
        
    def fit_transform_interactions(
        self, 
        interactions_df: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "item_id",
        rating_col: str = "rating"
    ) -> pd.DataFrame:
        """Fit encoders and transform interaction data.
        
        Args:
            interactions_df: Raw interactions DataFrame.
            user_col: Name of the user ID column.
            item_col: Name of the item ID column.
            rating_col: Name of the rating column.
            
        Returns:
            Processed interactions DataFrame with encoded IDs.
        """
        df = interactions_df.copy()
        
        # Encode user and item IDs
        df[f"{user_col}_encoded"] = self.user_encoder.fit_transform(df[user_col])
        df[f"{item_col}_encoded"] = self.item_encoder.fit_transform(df[item_col])
        
        # Ensure rating column exists and is numeric
        if rating_col not in df.columns:
            df[rating_col] = 1.0  # Implicit feedback
        else:
            df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce').fillna(1.0)
            
        self.is_fitted = True
        logger.info(f"Processed {len(df)} interactions")
        return df
        
    def transform_interactions(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        """Transform new interaction data using fitted encoders.
        
        Args:
            interactions_df: Raw interactions DataFrame.
            
        Returns:
            Processed interactions DataFrame with encoded IDs.
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before transforming new data")
            
        df = interactions_df.copy()
        df["user_id_encoded"] = self.user_encoder.transform(df["user_id"])
        df["item_id_encoded"] = self.item_encoder.transform(df["item_id"])
        return df
        
    def create_train_test_split(
        self,
        interactions_df: pd.DataFrame,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify_by_user: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train-test split for interactions.
        
        Args:
            interactions_df: Processed interactions DataFrame.
            test_size: Proportion of data to use for testing.
            random_state: Random seed for reproducibility.
            stratify_by_user: Whether to stratify by user for balanced splits.
            
        Returns:
            Tuple of (train_df, test_df).
        """
        if stratify_by_user:
            train_df, test_df = train_test_split(
                interactions_df,
                test_size=test_size,
                random_state=random_state,
                stratify=interactions_df["user_id_encoded"]
            )
        else:
            train_df, test_df = train_test_split(
                interactions_df,
                test_size=test_size,
                random_state=random_state
            )
            
        logger.info(f"Train set: {len(train_df)} interactions")
        logger.info(f"Test set: {len(test_df)} interactions")
        return train_df, test_df
        
    def create_user_item_matrix(
        self,
        interactions_df: pd.DataFrame,
        user_col: str = "user_id_encoded",
        item_col: str = "item_id_encoded",
        rating_col: str = "rating"
    ) -> np.ndarray:
        """Create user-item interaction matrix.
        
        Args:
            interactions_df: Processed interactions DataFrame.
            user_col: Name of the encoded user ID column.
            item_col: Name of the encoded item ID column.
            rating_col: Name of the rating column.
            
        Returns:
            User-item interaction matrix.
        """
        n_users = interactions_df[user_col].nunique()
        n_items = interactions_df[item_col].nunique()
        
        matrix = np.zeros((n_users, n_items))
        
        for _, row in interactions_df.iterrows():
            matrix[row[user_col], row[item_col]] = row[rating_col]
            
        logger.info(f"Created user-item matrix: {matrix.shape}")
        return matrix
        
    def get_user_item_indices(
        self,
        interactions_df: pd.DataFrame
    ) -> Tuple[List[int], List[int]]:
        """Get user and item indices for sparse matrix operations.
        
        Args:
            interactions_df: Processed interactions DataFrame.
            
        Returns:
            Tuple of (user_indices, item_indices).
        """
        user_indices = interactions_df["user_id_encoded"].values
        item_indices = interactions_df["item_id_encoded"].values
        return user_indices, item_indices
