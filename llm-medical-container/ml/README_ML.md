# Machine Learning Enhancement for Semantic Matching

## Overview

The ML enhancement system uses neural networks to improve semantic matching accuracy by learning from patterns in patient responses and successful/failed matches.

## Features

1. **Neural Network-Based Similarity Improvement**
   - Learns from successful and failed matches
   - Adjusts similarity scores based on learned patterns
   - Improves matching accuracy over time

2. **Pattern Recognition**
   - Recognizes common patient response patterns
   - Identifies similar phrasing across different patients
   - Learns domain-specific terminology patterns

3. **Adaptive Thresholds**
   - Dynamically adjusts matching thresholds per element/category
   - Adapts based on success rates
   - Optimizes for different medical specialties

4. **Continuous Learning**
   - Records successful and failed matches automatically
   - Trains models periodically on collected data
   - Improves performance over time

## Architecture

### Components

1. **SemanticMLEnhancer** (`ml/semantic_ml_enhancer.py`)
   - Main ML enhancement class
   - Manages neural network models
   - Handles training data collection and storage

2. **SemanticMatchingNN**
   - Neural network for improving similarity scores
   - Input: Base similarity + contextual features
   - Output: Adjusted similarity score (0-1)

3. **PatternRecognitionNN**
   - Neural network for pattern recognition
   - Input: Embedding features
   - Output: Pattern match probabilities

### Integration

The ML enhancer is integrated into `MedicalRuleEngine`:
- Automatically improves similarity scores in `compute_semantic_similarity()`
- Enhances FAISS matching in `find_matching_terms_faiss()`
- Adjusts thresholds dynamically
- Records matches for learning

## Usage

### Enable ML Learning

Set environment variable:
```bash
export ENABLE_ML_LEARNING=true
```

Or in code:
```python
from ml.medical_rule_engine import MedicalRuleEngine

engine = MedicalRuleEngine(
    embedding_model=embedding_model,
    enable_ml_learning=True
)
```

### Training Models

Train models on collected data:
```python
# Train similarity model
engine.train_ml_models(epochs=10)

# Get statistics
stats = engine.get_ml_stats()
print(f"Improved matches: {stats['improved_matches']}")
print(f"Training examples: {stats['training_examples']}")
```

### Recording Matches

Matches are automatically recorded when ML learning is enabled. You can also manually record:
```python
# Record successful match
engine.record_match_for_learning(
    patient_text="right side pain",
    guideline_text="right upper quadrant",
    element="location",
    similarity=0.75,
    was_successful=True,
    embedding=embedding_vector
)
```

### Adjusting Thresholds

Thresholds are automatically adjusted based on success rates. You can also manually adjust:
```python
# Adjust threshold for an element
engine.adjust_threshold_for_element("location", success_rate=0.85)
```

## Data Storage

All ML data is stored in `data/learning/`:
- `similarity_model.pth` - Trained similarity model
- `pattern_model.pth` - Trained pattern recognition model
- `training_data.json` - Collected training examples
- `learned_patterns.json` - Learned patterns
- `adaptive_thresholds.json` - Adaptive threshold values

## Features Extracted

The ML system extracts the following features for neural network input:

1. Base similarity score
2. Text length ratio (patient/guideline)
3. Word overlap ratio
4. Element type encoding (one-hot)
5. Category encoding (one-hot)
6. Embedding statistics (mean, std, max, min)
7. Character-level features
8. Word count features

## Training

### Automatic Training

Models are trained automatically when:
- Sufficient training data is collected (100+ examples)
- `train_ml_models()` is called manually

### Manual Training

```python
# Train with custom parameters
engine.train_ml_models(epochs=20)
```

### Training Data Requirements

- Minimum 100 examples for initial training
- Balanced positive/negative examples recommended
- More data = better performance

## Performance

### Benefits

- **Improved Accuracy**: Better matching through learned patterns
- **Adaptive Thresholds**: Optimized per element/category
- **Pattern Recognition**: Identifies common patient phrasing
- **Continuous Improvement**: Gets better over time

### Statistics

Track ML performance:
```python
stats = engine.get_ml_stats()
print(f"Total matches: {stats['total_matches']}")
print(f"Improved matches: {stats['improved_matches']}")
print(f"Patterns recognized: {stats['patterns_recognized']}")
print(f"Threshold adjustments: {stats['threshold_adjustments']}")
```

## Dependencies

- PyTorch (optional - ML features work without it, but disabled)
- NumPy
- JSON (standard library)

## Configuration

### Environment Variables

- `ENABLE_ML_LEARNING`: Enable/disable ML learning (default: false)
- `ENABLED_MEDICAL_CATEGORIES`: Categories to enable (affects ML training)

### Model Parameters

Default model architecture:
- Similarity Model: 10 input features → [64, 32, 16] hidden → 1 output
- Pattern Model: 384 embedding dim → [128, 64] hidden → 20 patterns

## Troubleshooting

### ML Not Working

1. Check PyTorch is installed: `pip install torch`
2. Verify `ENABLE_ML_LEARNING=true` is set
3. Check logs for initialization errors

### Low Performance

1. Train models with more data
2. Increase training epochs
3. Check training data quality
4. Verify sufficient examples collected

### Memory Issues

- Training data is limited to 10,000 successful matches and 5,000 failed matches
- Models are saved periodically
- Old training data is automatically pruned

## Future Enhancements

- Multi-task learning for different elements
- Transfer learning from pre-trained models
- Active learning for better data collection
- Real-time threshold adjustment
- Pattern visualization and analysis

