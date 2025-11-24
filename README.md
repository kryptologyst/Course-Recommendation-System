# Course Recommendation System

A comprehensive course recommendation system that implements multiple recommendation approaches including content-based filtering, collaborative filtering, and hybrid methods. The system provides a complete pipeline from data processing to model evaluation with an interactive web interface.

## Features

- **Multiple Recommendation Approaches**: Content-based, collaborative filtering, and hybrid methods
- **Comprehensive Evaluation**: Precision@K, Recall@K, NDCG@K, MAP@K, Hit Rate@K, Coverage, and Diversity metrics
- **Interactive Demo**: Streamlit-based web interface for exploring recommendations
- **Data Analysis**: Visualizations and insights about the dataset
- **Model Comparison**: Side-by-side performance comparison
- **Production Ready**: Clean code with type hints, comprehensive documentation, and testing

## Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or conda package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/Course-Recommendation-System.git
cd Course-Recommendation-System
```

2. Install dependencies:
```bash
pip install -e .
```

Or install development dependencies:
```bash
pip install -e ".[dev]"
```

### Generate Sample Data

The system includes a script to generate realistic sample data:

```bash
python scripts/generate_data.py
```

This will create:
- `data/raw/interactions.csv`: User-course interactions with ratings
- `data/raw/items.csv`: Course information including descriptions and metadata
- `data/raw/users.csv`: User profiles with preferences and experience levels

### Train Models

Train all recommendation models:

```bash
python train.py
```

Or train with custom configuration:

```bash
python train.py --config configs/config.yaml
```

### Run Interactive Demo

Launch the Streamlit demo:

```bash
streamlit run demo.py
```

The demo will be available at `http://localhost:8501`

## Project Structure

```
course-recommendation-system/
├── src/
│   └── course_recommender/
│       ├── __init__.py
│       ├── data/                 # Data loading and processing
│       │   ├── __init__.py
│       │   └── loaders.py
│       ├── models/               # Recommendation models
│       │   ├── __init__.py
│       │   ├── content_based.py
│       │   ├── collaborative.py
│       │   └── hybrid.py
│       ├── evaluation/           # Evaluation metrics and comparison
│       │   └── __init__.py
│       └── utils/               # Utility functions
│           └── __init__.py
├── data/
│   ├── raw/                     # Raw data files
│   └── processed/              # Processed data files
├── models/                      # Trained model files
├── configs/                     # Configuration files
├── scripts/                     # Utility scripts
├── tests/                       # Unit tests
├── notebooks/                   # Jupyter notebooks
├── assets/                      # Static assets
├── demo.py                      # Streamlit demo application
├── train.py                     # Main training script
├── pyproject.toml              # Project configuration
├── .gitignore                  # Git ignore file
└── README.md                   # This file
```

## Models

### Content-Based Filtering

Uses TF-IDF vectorization of course descriptions and user profiles to recommend courses based on content similarity.

**Features:**
- TF-IDF vectorization with configurable parameters
- Optional sentence transformer embeddings
- User profile creation from interaction history
- Similarity-based recommendations

### Collaborative Filtering

Implements matrix factorization to find latent user-item relationships and recommend courses based on similar users' preferences.

**Features:**
- Stochastic gradient descent optimization
- User and item bias terms
- Configurable number of factors and iterations
- Cold-start handling

### Hybrid Approach

Combines content-based and collaborative filtering methods for improved recommendations.

**Features:**
- Weighted averaging of recommendation scores
- Optional machine learning blending
- Explanation generation for recommendations
- Configurable weights for different approaches

## Evaluation Metrics

The system provides comprehensive evaluation using multiple metrics:

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended  
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision
- **Hit Rate@K**: Fraction of users for whom at least one relevant item is recommended
- **Coverage**: Fraction of catalog items that are recommended
- **Diversity**: Intra-list diversity of recommendations

## Configuration

The system uses YAML configuration files for easy customization. Key configuration options:

```yaml
# Data configuration
data:
  data_dir: "data/raw"
  test_size: 0.2
  random_state: 42

# Model configurations
models:
  content_based:
    use_sentence_transformers: false
    max_features: 1000
    
  collaborative:
    n_factors: 50
    n_iterations: 20
    
  hybrid:
    content_weight: 0.5
    collaborative_weight: 0.5

# Evaluation configuration
evaluation:
  k_values: [5, 10, 20]
  n_recommendations: 20
```

## Usage Examples

### Basic Usage

```python
from course_recommender import DataLoader, ContentBasedRecommender

# Load data
loader = DataLoader("data/raw")
interactions_df = loader.load_interactions("interactions.csv")
items_df = loader.load_items("items.csv")

# Train model
model = ContentBasedRecommender()
model.fit(items_df)

# Get recommendations
recommendations = model.recommend(
    user_id="user_0001",
    user_interactions=interactions_df[interactions_df["user_id"] == "user_0001"],
    items_df=items_df,
    n_recommendations=10
)
```

### Model Comparison

```python
from course_recommender import Evaluator

# Evaluate models
evaluator = Evaluator(k_values=[5, 10, 20])
results = evaluator.compare_models(
    models={"content": content_model, "collaborative": collab_model},
    test_interactions=test_df,
    items_df=items_df,
    users_df=users_df
)
```

## Dataset Schema

### Interactions (`interactions.csv`)
- `user_id`: Unique user identifier
- `item_id`: Unique course identifier  
- `rating`: User rating (1-5 scale)
- `timestamp`: Interaction timestamp
- `interaction_type`: Type of interaction (enrollment, completion, review, bookmark)
- `duration_minutes`: Time spent (optional)

### Items (`items.csv`)
- `item_id`: Unique course identifier
- `title`: Course title
- `description`: Course description
- `category`: Course category (Programming, Data Science, etc.)
- `difficulty`: Difficulty level (1-5)
- `duration_hours`: Course duration in hours
- `prerequisites`: Required prerequisites
- `tags`: Course tags (pipe-separated)
- `price`: Course price
- `rating`: Average course rating
- `enrollment_count`: Number of enrollments

### Users (`users.csv`)
- `user_id`: Unique user identifier
- `age`: User age
- `experience_level`: Experience level (beginner, intermediate, advanced)
- `primary_interest`: Primary learning interest
- `learning_style`: Preferred learning style
- `preferred_difficulty`: Preferred difficulty level
- `max_course_duration`: Maximum preferred course duration
- `budget_preference`: Budget preference (free, low, medium, high)

## Development

### Code Quality

The project uses modern Python development practices:

- **Type Hints**: All functions include type annotations
- **Documentation**: Google-style docstrings throughout
- **Code Formatting**: Black for code formatting
- **Linting**: Ruff for code linting
- **Testing**: pytest for unit tests

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
ruff check src/ tests/
```

### Pre-commit Hooks

Install pre-commit hooks:

```bash
pre-commit install
```

## Performance

The system is designed for efficiency and scalability:

- **Caching**: Streamlit demo uses caching for faster loading
- **Vectorization**: NumPy operations for numerical computations
- **Sparse Matrices**: Efficient handling of user-item matrices
- **Batch Processing**: Efficient batch operations for large datasets

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- scikit-learn for machine learning utilities
- Streamlit for the interactive web interface
- Plotly for interactive visualizations
- The open-source community for various libraries and tools

## Future Enhancements

- Real-time recommendation updates
- Advanced deep learning models (Neural Collaborative Filtering, DeepFM)
- Multi-criteria recommendations
- Explainable AI features
- A/B testing framework
- REST API for production deployment
- Docker containerization
- Cloud deployment configurations
# Course-Recommendation-System
