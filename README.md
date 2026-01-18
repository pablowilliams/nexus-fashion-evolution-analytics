# NEXUS: Fashion Evolution Analytics

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Machine learning analysis of fashion trend evolution from 2015-2024, predicting trend scores and identifying the factors driving style changes including the rise of oversized silhouettes and sustainable materials.

## Key Results

| Metric | Value |
|--------|-------|
| R² Score | 0.82 |
| MAE | 7.8 trend points |
| Key Finding | Social media mentions are the #1 predictor of trend scores |

## Dataset

`fashion_evolution.csv` contains 18,000 fashion items with 28 features:
- Temporal: year, quarter, time_period
- Attributes: category, silhouette, neckline, sleeve_type, hemline, color, pattern, fabric
- Sustainability: sustainability_label (Conventional/Organic/Recycled/Sustainable/Upcycled)
- Engagement: search_volume, social_mentions, runway_appearances, celebrity_wears, influencer_posts
- Performance: units_sold, return_rate, avg_rating, trend_score

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src.data_loader import DataLoader
from src.trend_analyzer import TrendAnalyzer
from src.predictor import TrendPredictor

loader = DataLoader('data/fashion_evolution.csv')
df = loader.load_data()

analyzer = TrendAnalyzer()
rising_silhouettes = analyzer.identify_rising_trends(df, 'silhouette')

predictor = TrendPredictor()
predictor.train(X_train, y_train)
predictions = predictor.predict(X_test)
```

## License

MIT License
