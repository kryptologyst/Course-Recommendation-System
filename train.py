"""
Main training script for the course recommendation system.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from course_recommender.data import DataLoader, DataProcessor
from course_recommender.models import ContentBasedRecommender, CollaborativeRecommender, HybridRecommender
from course_recommender.evaluation import Evaluator
from course_recommender.utils import set_seed, load_config, save_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """Train all recommendation models.
    
    Args:
        config: Configuration dictionary.
        
    Returns:
        Dictionary containing trained models.
    """
    logger.info("Starting model training...")
    
    # Set random seed
    set_seed(config["training"]["random_state"])
    
    # Load data
    logger.info("Loading data...")
    loader = DataLoader(config["data"]["data_dir"])
    
    interactions_df = loader.load_interactions(config["data"]["interactions_file"])
    items_df = loader.load_items(config["data"]["items_file"])
    users_df = loader.load_users(config["data"]["users_file"])
    
    logger.info(f"Loaded {len(interactions_df)} interactions, {len(items_df)} items, {len(users_df)} users")
    
    # Process data
    logger.info("Processing data...")
    processor = DataProcessor()
    processed_interactions = processor.fit_transform_interactions(interactions_df)
    
    # Split data
    train_df, test_df = processor.create_train_test_split(
        processed_interactions,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"]
    )
    
    logger.info(f"Train set: {len(train_df)} interactions")
    logger.info(f"Test set: {len(test_df)} interactions")
    
    # Initialize models
    models = {}
    
    # Content-based model
    logger.info("Training content-based model...")
    content_model = ContentBasedRecommender(
        use_sentence_transformers=config["models"]["content_based"]["use_sentence_transformers"],
        model_name=config["models"]["content_based"]["model_name"],
        max_features=config["models"]["content_based"]["max_features"],
        ngram_range=tuple(config["models"]["content_based"]["ngram_range"])
    )
    content_model.fit(items_df)
    models["content_based"] = content_model
    
    # Collaborative filtering model
    logger.info("Training collaborative filtering model...")
    collaborative_model = CollaborativeRecommender(
        n_factors=config["models"]["collaborative"]["n_factors"],
        n_iterations=config["models"]["collaborative"]["n_iterations"],
        learning_rate=config["models"]["collaborative"]["learning_rate"],
        regularization=config["models"]["collaborative"]["regularization"],
        random_state=config["models"]["collaborative"]["random_state"]
    )
    collaborative_model.fit(train_df)
    models["collaborative"] = collaborative_model
    
    # Hybrid model
    logger.info("Training hybrid model...")
    hybrid_model = HybridRecommender(
        content_weight=config["models"]["hybrid"]["content_weight"],
        collaborative_weight=config["models"]["hybrid"]["collaborative_weight"],
        use_ml_blending=config["models"]["hybrid"]["use_ml_blending"],
        blending_model=config["models"]["hybrid"]["blending_model"]
    )
    hybrid_model.fit(train_df, items_df, users_df)
    models["hybrid"] = hybrid_model
    
    logger.info("Model training completed successfully")
    
    return models, train_df, test_df, items_df, users_df


def evaluate_models(
    models: Dict[str, Any],
    test_df: Any,
    items_df: Any,
    users_df: Any,
    config: Dict[str, Any]
) -> None:
    """Evaluate trained models.
    
    Args:
        models: Dictionary containing trained models.
        test_df: Test dataset.
        items_df: Items dataset.
        users_df: Users dataset.
        config: Configuration dictionary.
    """
    logger.info("Starting model evaluation...")
    
    evaluator = Evaluator(k_values=config["evaluation"]["k_values"])
    
    # Evaluate each model
    results = {}
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name} model...")
        results[model_name] = evaluator.evaluate_model(
            model,
            test_df,
            items_df,
            users_df,
            n_recommendations=config["evaluation"]["n_recommendations"],
            model_name=model_name
        )
    
    # Create comparison DataFrame
    comparison_data = []
    for result in results.values():
        row = {"Model": result["model_name"].title()}
        row.update(result["metrics"])
        comparison_data.append(row)
    
    import pandas as pd
    comparison_df = pd.DataFrame(comparison_data)
    
    # Display results
    logger.info("Evaluation Results:")
    logger.info(f"\n{comparison_df.to_string(index=False)}")
    
    # Generate report
    report = evaluator.generate_report(comparison_df)
    logger.info(f"\n{report}")
    
    # Save results
    os.makedirs("results", exist_ok=True)
    comparison_df.to_csv("results/model_comparison.csv", index=False)
    
    with open("results/evaluation_report.txt", "w") as f:
        f.write(report)
    
    logger.info("Evaluation results saved to results/ directory")


def save_models(models: Dict[str, Any], config: Dict[str, Any]) -> None:
    """Save trained models.
    
    Args:
        models: Dictionary containing trained models.
        config: Configuration dictionary.
    """
    if not config["training"]["save_models"]:
        return
        
    logger.info("Saving models...")
    
    model_dir = Path(config["training"]["model_dir"])
    model_dir.mkdir(exist_ok=True)
    
    import pickle
    
    for model_name, model in models.items():
        model_path = model_dir / f"{model_name}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        logger.info(f"Saved {model_name} model to {model_path}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train course recommendation models")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only evaluate existing models"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation after training"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    if not args.skip_training:
        # Train models
        models, train_df, test_df, items_df, users_df = train_models(config)
        
        # Save models
        save_models(models, config)
    else:
        # Load existing models
        logger.info("Loading existing models...")
        import pickle
        
        models = {}
        model_dir = Path(config["training"]["model_dir"])
        
        for model_name in ["content_based", "collaborative", "hybrid"]:
            model_path = model_dir / f"{model_name}_model.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    models[model_name] = pickle.load(f)
                logger.info(f"Loaded {model_name} model")
            else:
                logger.warning(f"Model file not found: {model_path}")
        
        if not models:
            logger.error("No models found. Please train models first.")
            return
            
        # Load data for evaluation
        loader = DataLoader(config["data"]["data_dir"])
        interactions_df = loader.load_interactions(config["data"]["interactions_file"])
        items_df = loader.load_items(config["data"]["items_file"])
        users_df = loader.load_users(config["data"]["users_file"])
        
        processor = DataProcessor()
        processed_interactions = processor.fit_transform_interactions(interactions_df)
        _, test_df = processor.create_train_test_split(
            processed_interactions,
            test_size=config["data"]["test_size"],
            random_state=config["data"]["random_state"]
        )
    
    if not args.skip_evaluation:
        # Evaluate models
        evaluate_models(models, test_df, items_df, users_df, config)
    
    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()
