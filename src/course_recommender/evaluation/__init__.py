"""
Evaluation metrics and model comparison utilities.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable
from sklearn.metrics import precision_score, recall_score
import logging

logger = logging.getLogger(__name__)


class Metrics:
    """Collection of recommendation evaluation metrics."""
    
    @staticmethod
    def precision_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Precision@K.
        
        Args:
            recommended_items: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            Precision@K score.
        """
        if k == 0:
            return 0.0
            
        top_k_recommendations = recommended_items[:k]
        relevant_in_top_k = len(set(top_k_recommendations) & set(relevant_items))
        
        return relevant_in_top_k / k
        
    @staticmethod
    def recall_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Recall@K.
        
        Args:
            recommended_items: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            Recall@K score.
        """
        if len(relevant_items) == 0:
            return 0.0
            
        top_k_recommendations = recommended_items[:k]
        relevant_in_top_k = len(set(top_k_recommendations) & set(relevant_items))
        
        return relevant_in_top_k / len(relevant_items)
        
    @staticmethod
    def ndcg_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate NDCG@K.
        
        Args:
            recommended_items: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            NDCG@K score.
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0
            
        top_k_recommendations = recommended_items[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, item in enumerate(top_k_recommendations):
            if item in relevant_items:
                dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1) = 0
                
        # Calculate IDCG (ideal DCG)
        idcg = 0.0
        for i in range(min(k, len(relevant_items))):
            idcg += 1.0 / np.log2(i + 2)
            
        return dcg / idcg if idcg > 0 else 0.0
        
    @staticmethod
    def map_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate MAP@K (Mean Average Precision).
        
        Args:
            recommended_items: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            MAP@K score.
        """
        if k == 0 or len(relevant_items) == 0:
            return 0.0
            
        top_k_recommendations = recommended_items[:k]
        
        precision_sum = 0.0
        relevant_count = 0
        
        for i, item in enumerate(top_k_recommendations):
            if item in relevant_items:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precision_sum += precision_at_i
                
        return precision_sum / len(relevant_items) if len(relevant_items) > 0 else 0.0
        
    @staticmethod
    def hit_rate_at_k(recommended_items: List[str], relevant_items: List[str], k: int) -> float:
        """Calculate Hit Rate@K.
        
        Args:
            recommended_items: List of recommended item IDs.
            relevant_items: List of relevant item IDs.
            k: Number of top recommendations to consider.
            
        Returns:
            Hit Rate@K score (1 if any relevant item in top-k, 0 otherwise).
        """
        if k == 0:
            return 0.0
            
        top_k_recommendations = recommended_items[:k]
        return 1.0 if len(set(top_k_recommendations) & set(relevant_items)) > 0 else 0.0
        
    @staticmethod
    def coverage(recommended_items_all_users: List[List[str]], all_items: List[str]) -> float:
        """Calculate catalog coverage.
        
        Args:
            recommended_items_all_users: List of recommendations for all users.
            all_items: List of all available items.
            
        Returns:
            Coverage score.
        """
        if len(all_items) == 0:
            return 0.0
            
        recommended_items_set = set()
        for user_recommendations in recommended_items_all_users:
            recommended_items_set.update(user_recommendations)
            
        return len(recommended_items_set) / len(all_items)
        
    @staticmethod
    def diversity(recommended_items: List[str], item_features: Optional[np.ndarray] = None) -> float:
        """Calculate intra-list diversity.
        
        Args:
            recommended_items: List of recommended item IDs.
            item_features: Optional feature matrix for items.
            
        Returns:
            Diversity score.
        """
        if len(recommended_items) <= 1:
            return 0.0
            
        if item_features is not None:
            # Calculate average pairwise distance
            distances = []
            for i in range(len(recommended_items)):
                for j in range(i + 1, len(recommended_items)):
                    # This is a simplified version - in practice, you'd need item indices
                    distances.append(1.0)  # Placeholder
            return np.mean(distances) if distances else 0.0
        else:
            # Simple diversity based on unique items
            return len(set(recommended_items)) / len(recommended_items)


class Evaluator:
    """Comprehensive evaluation system for recommendation models."""
    
    def __init__(self, k_values: List[int] = [5, 10, 20]):
        """Initialize the evaluator.
        
        Args:
            k_values: List of k values for evaluation metrics.
        """
        self.k_values = k_values
        self.metrics = Metrics()
        
    def evaluate_model(
        self,
        model: Any,
        test_interactions: pd.DataFrame,
        items_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None,
        n_recommendations: int = 20,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Evaluate a recommendation model.
        
        Args:
            model: Recommendation model to evaluate.
            test_interactions: DataFrame containing test interactions.
            items_df: DataFrame containing item information.
            users_df: Optional DataFrame containing user information.
            n_recommendations: Number of recommendations to generate per user.
            model_name: Name of the model for reporting.
            
        Returns:
            Dictionary containing evaluation results.
        """
        logger.info(f"Evaluating {model_name}...")
        
        results = {
            "model_name": model_name,
            "k_values": self.k_values,
            "metrics": {}
        }
        
        # Initialize metric accumulators
        precision_scores = {k: [] for k in self.k_values}
        recall_scores = {k: [] for k in self.k_values}
        ndcg_scores = {k: [] for k in self.k_values}
        map_scores = {k: [] for k in self.k_values}
        hit_rate_scores = {k: [] for k in self.k_values}
        
        all_recommendations = []
        all_items = items_df["item_id"].tolist()
        
        # Evaluate for each user
        unique_users = test_interactions["user_id"].unique()
        
        for user_id in unique_users:
            user_test_items = test_interactions[test_interactions["user_id"] == user_id]["item_id"].tolist()
            
            if len(user_test_items) == 0:
                continue
                
            # Get user interactions for training (if needed)
            user_interactions = test_interactions[test_interactions["user_id"] == user_id]
            
            try:
                # Generate recommendations
                if hasattr(model, 'recommend'):
                    recommendations = model.recommend(
                        user_id=user_id,
                        interactions_df=user_interactions,
                        items_df=items_df,
                        users_df=users_df,
                        n_recommendations=n_recommendations,
                        exclude_interacted=True
                    )
                    recommended_items = [item_id for item_id, _ in recommendations]
                else:
                    logger.warning(f"Model {model_name} does not have recommend method")
                    continue
                    
            except Exception as e:
                logger.warning(f"Error generating recommendations for user {user_id}: {e}")
                continue
                
            if len(recommended_items) == 0:
                continue
                
            all_recommendations.append(recommended_items)
            
            # Calculate metrics for each k
            for k in self.k_values:
                precision_scores[k].append(
                    self.metrics.precision_at_k(recommended_items, user_test_items, k)
                )
                recall_scores[k].append(
                    self.metrics.recall_at_k(recommended_items, user_test_items, k)
                )
                ndcg_scores[k].append(
                    self.metrics.ndcg_at_k(recommended_items, user_test_items, k)
                )
                map_scores[k].append(
                    self.metrics.map_at_k(recommended_items, user_test_items, k)
                )
                hit_rate_scores[k].append(
                    self.metrics.hit_rate_at_k(recommended_items, user_test_items, k)
                )
                
        # Calculate average metrics
        for k in self.k_values:
            results["metrics"][f"precision@{k}"] = np.mean(precision_scores[k])
            results["metrics"][f"recall@{k}"] = np.mean(recall_scores[k])
            results["metrics"][f"ndcg@{k}"] = np.mean(ndcg_scores[k])
            results["metrics"][f"map@{k}"] = np.mean(map_scores[k])
            results["metrics"][f"hit_rate@{k}"] = np.mean(hit_rate_scores[k])
            
        # Calculate coverage
        results["metrics"]["coverage"] = self.metrics.coverage(all_recommendations, all_items)
        
        # Calculate diversity (simplified)
        results["metrics"]["diversity"] = self.metrics.diversity(
            [item for recs in all_recommendations for item in recs]
        )
        
        logger.info(f"Evaluation completed for {model_name}")
        return results
        
    def compare_models(
        self,
        models: Dict[str, Any],
        test_interactions: pd.DataFrame,
        items_df: pd.DataFrame,
        users_df: Optional[pd.DataFrame] = None,
        n_recommendations: int = 20
    ) -> pd.DataFrame:
        """Compare multiple recommendation models.
        
        Args:
            models: Dictionary mapping model names to model instances.
            test_interactions: DataFrame containing test interactions.
            items_df: DataFrame containing item information.
            users_df: Optional DataFrame containing user information.
            n_recommendations: Number of recommendations to generate per user.
            
        Returns:
            DataFrame containing comparison results.
        """
        logger.info("Comparing multiple models...")
        
        all_results = []
        
        for model_name, model in models.items():
            results = self.evaluate_model(
                model, test_interactions, items_df, users_df, n_recommendations, model_name
            )
            all_results.append(results)
            
        # Create comparison DataFrame
        comparison_data = []
        
        for result in all_results:
            row = {"Model": result["model_name"]}
            row.update(result["metrics"])
            comparison_data.append(row)
            
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by a primary metric (e.g., NDCG@10)
        if "ndcg@10" in comparison_df.columns:
            comparison_df = comparison_df.sort_values("ndcg@10", ascending=False)
            
        logger.info("Model comparison completed")
        return comparison_df
        
    def generate_report(self, comparison_df: pd.DataFrame) -> str:
        """Generate a formatted evaluation report.
        
        Args:
            comparison_df: DataFrame containing model comparison results.
            
        Returns:
            Formatted report string.
        """
        report = "=" * 80 + "\n"
        report += "RECOMMENDATION MODEL EVALUATION REPORT\n"
        report += "=" * 80 + "\n\n"
        
        # Summary statistics
        report += "SUMMARY STATISTICS:\n"
        report += f"Number of models compared: {len(comparison_df)}\n"
        report += f"Evaluation metrics: {len(comparison_df.columns) - 1}\n\n"
        
        # Top performing model
        if len(comparison_df) > 0:
            best_model = comparison_df.iloc[0]
            report += f"BEST PERFORMING MODEL: {best_model['Model']}\n"
            report += f"NDCG@10: {best_model.get('ndcg@10', 'N/A'):.4f}\n"
            report += f"Precision@10: {best_model.get('precision@10', 'N/A'):.4f}\n"
            report += f"Recall@10: {best_model.get('recall@10', 'N/A'):.4f}\n\n"
            
        # Detailed results
        report += "DETAILED RESULTS:\n"
        report += "-" * 80 + "\n"
        
        # Format the DataFrame for better readability
        formatted_df = comparison_df.copy()
        for col in formatted_df.columns:
            if col != "Model" and isinstance(formatted_df[col].iloc[0], float):
                formatted_df[col] = formatted_df[col].round(4)
                
        report += formatted_df.to_string(index=False)
        report += "\n\n"
        
        # Recommendations
        report += "RECOMMENDATIONS:\n"
        report += "-" * 40 + "\n"
        
        if len(comparison_df) > 0:
            best_model_name = comparison_df.iloc[0]["Model"]
            report += f"• Use {best_model_name} for best overall performance\n"
            
            # Check for specific strengths
            if "coverage" in comparison_df.columns:
                best_coverage_idx = comparison_df["coverage"].idxmax()
                if best_coverage_idx != comparison_df.index[0]:
                    best_coverage_model = comparison_df.loc[best_coverage_idx, "Model"]
                    report += f"• Use {best_coverage_model} for best catalog coverage\n"
                    
        report += "• Consider ensemble methods for production deployment\n"
        report += "• Monitor performance on new data regularly\n"
        
        return report
