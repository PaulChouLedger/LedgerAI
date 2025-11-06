#!/usr/bin/env python3
"""
Semantic ML Enhancer - Neural Network for Pattern Recognition and Improved Semantic Matching

This module uses neural networks to:
1. Learn patterns from successful/failed semantic matches
2. Improve similarity scores based on learned patterns
3. Adapt thresholds dynamically per element/category
4. Recognize common patient response patterns
5. Improve matching accuracy over time
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict
import pickle
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[ML] ⚠️ PyTorch not available. ML enhancements will be disabled.")


class SemanticMatchDataset(Dataset):
    """Dataset for training semantic matching neural network"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class SemanticMatchingNN(nn.Module):
    """
    Neural Network for improving semantic similarity scores
    
    Architecture:
    - Input: Base similarity score + contextual features
    - Hidden layers: Learn complex patterns
    - Output: Adjusted similarity score (0-1)
    """
    
    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = [64, 32, 16], dropout: float = 0.2):
        super(SemanticMatchingNN, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
        
        # Output layer (single value: adjusted similarity)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Ensure output is 0-1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x).squeeze()


class PatternRecognitionNN(nn.Module):
    """
    Neural Network for recognizing patterns in patient responses
    
    Architecture:
    - Input: Embedding features + context
    - Hidden layers: Pattern recognition
    - Output: Pattern match probability
    """
    
    def __init__(self, embedding_dim: int = 384, hidden_dims: List[int] = [128, 64], num_patterns: int = 20):
        super(PatternRecognitionNN, self).__init__()
        
        layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # Output layer: pattern probabilities
        layers.append(nn.Linear(prev_dim, num_patterns))
        layers.append(nn.Softmax(dim=1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class SemanticMLEnhancer:
    """
    ML Enhancer for improving semantic matching using neural networks
    """
    
    def __init__(self, data_dir: Optional[str] = None, enable_learning: bool = True):
        """
        Initialize ML Enhancer
        
        Args:
            data_dir: Directory to store learned models and data
            enable_learning: Whether to enable learning from interactions
        """
        self.enable_learning = enable_learning and TORCH_AVAILABLE
        
        # Set up data directory
        if data_dir is None:
            current_file = Path(__file__).resolve()
            self.data_dir = current_file.parent.parent / 'data' / 'learning'
        else:
            self.data_dir = Path(data_dir)
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Model paths
        self.similarity_model_path = self.data_dir / 'similarity_model.pth'
        self.pattern_model_path = self.data_dir / 'pattern_model.pth'
        self.training_data_path = self.data_dir / 'training_data.json'
        self.patterns_path = self.data_dir / 'learned_patterns.json'
        self.thresholds_path = self.data_dir / 'adaptive_thresholds.json'
        
        # Initialize models
        self.similarity_model = None
        self.pattern_model = None
        self.optimizer_similarity = None
        self.optimizer_pattern = None
        
        # Training data storage
        self.training_data = {
            'similarity_matches': [],  # Successful matches for training
            'failed_matches': [],      # Failed matches for negative learning
            'pattern_examples': [],    # Pattern examples
            'threshold_adjustments': [] # Threshold adjustment history
        }
        
        # Learned patterns and adaptive thresholds
        self.learned_patterns = {}
        self.adaptive_thresholds = defaultdict(lambda: {
            'location': 0.65,
            'character': 0.70,
            'aggravating': 0.65,
            'relieving': 0.65,
            'timing': 0.70,
            'duration': 0.70,
            'severity': 0.75,
            'onset': 0.70
        })
        
        # Statistics
        self.stats = {
            'total_matches': 0,
            'improved_matches': 0,
            'patterns_recognized': 0,
            'threshold_adjustments': 0
        }
        
        # Load existing models and data
        if self.enable_learning:
            self._load_models()
            self._load_training_data()
            self._load_learned_patterns()
            self._load_adaptive_thresholds()
    
    def _load_models(self):
        """Load trained models if they exist"""
        if not TORCH_AVAILABLE:
            return
        
        try:
            # Load similarity adjustment model
            if self.similarity_model_path.exists():
                self.similarity_model = SemanticMatchingNN()
                self.similarity_model.load_state_dict(torch.load(self.similarity_model_path))
                self.similarity_model.eval()
                self.optimizer_similarity = optim.Adam(self.similarity_model.parameters(), lr=0.001)
                print("[ML] ✅ Loaded similarity adjustment model")
            
            # Load pattern recognition model
            if self.pattern_model_path.exists():
                self.pattern_model = PatternRecognitionNN()
                self.pattern_model.load_state_dict(torch.load(self.pattern_model_path))
                self.pattern_model.eval()
                self.optimizer_pattern = optim.Adam(self.pattern_model.parameters(), lr=0.001)
                print("[ML] ✅ Loaded pattern recognition model")
        except Exception as e:
            print(f"[ML] ⚠️ Error loading models: {e}")
    
    def _load_training_data(self):
        """Load training data from disk"""
        if self.training_data_path.exists():
            try:
                with open(self.training_data_path, 'r') as f:
                    self.training_data = json.load(f)
                print(f"[ML] ✅ Loaded {len(self.training_data['similarity_matches'])} training examples")
            except Exception as e:
                print(f"[ML] ⚠️ Error loading training data: {e}")
    
    def _load_learned_patterns(self):
        """Load learned patterns from disk"""
        if self.patterns_path.exists():
            try:
                with open(self.patterns_path, 'r') as f:
                    self.learned_patterns = json.load(f)
                print(f"[ML] ✅ Loaded {len(self.learned_patterns)} learned patterns")
            except Exception as e:
                print(f"[ML] ⚠️ Error loading patterns: {e}")
    
    def _load_adaptive_thresholds(self):
        """Load adaptive thresholds from disk"""
        if self.thresholds_path.exists():
            try:
                with open(self.thresholds_path, 'r') as f:
                    loaded = json.load(f)
                    for key, value in loaded.items():
                        self.adaptive_thresholds[key] = value
                print(f"[ML] ✅ Loaded adaptive thresholds for {len(self.adaptive_thresholds)} categories")
            except Exception as e:
                print(f"[ML] ⚠️ Error loading thresholds: {e}")
    
    def _save_models(self):
        """Save trained models to disk"""
        if not TORCH_AVAILABLE or not self.enable_learning:
            return
        
        try:
            if self.similarity_model:
                torch.save(self.similarity_model.state_dict(), self.similarity_model_path)
            if self.pattern_model:
                torch.save(self.pattern_model.state_dict(), self.pattern_model_path)
        except Exception as e:
            print(f"[ML] ⚠️ Error saving models: {e}")
    
    def _save_training_data(self):
        """Save training data to disk"""
        if not self.enable_learning:
            return
        
        try:
            with open(self.training_data_path, 'w') as f:
                json.dump(self.training_data, f, indent=2)
        except Exception as e:
            print(f"[ML] ⚠️ Error saving training data: {e}")
    
    def _save_learned_patterns(self):
        """Save learned patterns to disk"""
        if not self.enable_learning:
            return
        
        try:
            with open(self.patterns_path, 'w') as f:
                json.dump(self.learned_patterns, f, indent=2)
        except Exception as e:
            print(f"[ML] ⚠️ Error saving patterns: {e}")
    
    def _save_adaptive_thresholds(self):
        """Save adaptive thresholds to disk"""
        if not self.enable_learning:
            return
        
        try:
            with open(self.thresholds_path, 'w') as f:
                json.dump(dict(self.adaptive_thresholds), f, indent=2)
        except Exception as e:
            print(f"[ML] ⚠️ Error saving thresholds: {e}")
    
    def extract_features(self, base_similarity: float, patient_text: str, guideline_text: str,
                        element: str, category: str, embedding: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract features for neural network input
        
        Features:
        1. Base similarity score
        2. Text length ratio (patient/guideline)
        3. Word overlap ratio
        4. Element type encoding (one-hot)
        5. Category encoding (one-hot)
        6. Embedding statistics (mean, std, max, min) if available
        7. Character-level features
        8. Word count features
        """
        features = []
        
        # 1. Base similarity
        features.append(base_similarity)
        
        # 2. Text length ratio
        patient_len = len(patient_text)
        guideline_len = len(guideline_text)
        length_ratio = patient_len / max(guideline_len, 1)
        features.append(min(length_ratio, 2.0))  # Cap at 2.0
        
        # 3. Word overlap ratio
        patient_words = set(patient_text.lower().split())
        guideline_words = set(guideline_text.lower().split())
        if guideline_words:
            overlap = len(patient_words & guideline_words) / len(guideline_words)
        else:
            overlap = 0.0
        features.append(overlap)
        
        # 4. Element type encoding (one-hot: 8 elements)
        element_map = {
            'location': 0, 'character': 1, 'aggravating': 2, 'relieving': 3,
            'timing': 4, 'duration': 5, 'severity': 6, 'onset': 7
        }
        element_encoding = [0.0] * 8
        if element in element_map:
            element_encoding[element_map[element]] = 1.0
        features.extend(element_encoding)
        
        # 5. Category encoding (one-hot: 9 categories)
        category_map = {
            'gastrointestinal': 0, 'cardiovascular': 1, 'respiratory': 2,
            'neurological': 3, 'musculoskeletal': 4, 'renal': 5,
            'genitourinary': 6, 'gynecological': 7, 'dermatological': 8
        }
        category_encoding = [0.0] * 9
        if category in category_map:
            category_encoding[category_map[category]] = 1.0
        features.extend(category_encoding)
        
        # 6. Embedding statistics (if available)
        if embedding is not None:
            features.append(float(np.mean(embedding)))
            features.append(float(np.std(embedding)))
            features.append(float(np.max(embedding)))
            features.append(float(np.min(embedding)))
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 7. Character-level features
        features.append(len(patient_text) / 100.0)  # Normalized length
        features.append(patient_text.count(' ') / max(len(patient_text), 1))  # Space ratio
        
        # 8. Word count features
        patient_word_count = len(patient_text.split())
        guideline_word_count = len(guideline_text.split())
        features.append(patient_word_count / max(guideline_word_count, 1))
        
        return np.array(features, dtype=np.float32)
    
    def improve_similarity_score(self, base_similarity: float, patient_text: str, guideline_text: str,
                                 element: str, category: str, embedding: Optional[np.ndarray] = None) -> float:
        """
        Use neural network to improve similarity score
        
        Args:
            base_similarity: Base similarity score from FAISS/embedding
            patient_text: Patient response text
            guideline_text: Guideline term text
            element: OLDCARTS element type
            category: Medical category
            embedding: Optional embedding vector for additional features
        
        Returns:
            Improved similarity score (0-1)
        """
        if not self.enable_learning or self.similarity_model is None:
            return base_similarity
        
        try:
            # Extract features
            features = self.extract_features(
                base_similarity, patient_text, guideline_text, element, category, embedding
            )
            
            # Ensure we have the right input dimension
            if features.shape[0] < 10:
                # Pad with zeros if needed
                features = np.pad(features, (0, max(0, 10 - features.shape[0])), 'constant')
            elif features.shape[0] > 10:
                # Truncate if needed
                features = features[:10]
            
            # Predict improved score
            with torch.no_grad():
                input_tensor = torch.FloatTensor(features).unsqueeze(0)
                improved_score = self.similarity_model(input_tensor).item()
            
            # Track improvement
            if improved_score > base_similarity:
                self.stats['improved_matches'] += 1
            
            self.stats['total_matches'] += 1
            
            return float(np.clip(improved_score, 0.0, 1.0))
        except Exception as e:
            print(f"[ML] ⚠️ Error improving similarity: {e}")
            return base_similarity
    
    def recognize_pattern(self, patient_text: str, element: str, embedding: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Recognize patterns in patient response
        
        Args:
            patient_text: Patient response text
            element: OLDCARTS element type
            embedding: Optional embedding vector
        
        Returns:
            Dictionary of pattern probabilities
        """
        if not self.enable_learning or self.pattern_model is None or embedding is None:
            return {}
        
        try:
            # Use embedding as input (pad/truncate to expected dimension)
            if embedding.shape[0] < 384:
                embedding = np.pad(embedding, (0, 384 - embedding.shape[0]), 'constant')
            elif embedding.shape[0] > 384:
                embedding = embedding[:384]
            
            with torch.no_grad():
                input_tensor = torch.FloatTensor(embedding).unsqueeze(0)
                pattern_probs = self.pattern_model(input_tensor).squeeze().numpy()
            
            # Convert to dictionary
            patterns = {}
            for i, prob in enumerate(pattern_probs):
                if prob > 0.1:  # Only return significant patterns
                    patterns[f'pattern_{i}'] = float(prob)
            
            if patterns:
                self.stats['patterns_recognized'] += 1
            
            return patterns
        except Exception as e:
            print(f"[ML] ⚠️ Error recognizing pattern: {e}")
            return {}
    
    def get_adaptive_threshold(self, element: str, category: str) -> float:
        """
        Get adaptive threshold for element/category combination
        
        Args:
            element: OLDCARTS element type
            category: Medical category
        
        Returns:
            Adaptive threshold value
        """
        category_key = category or 'default'
        return self.adaptive_thresholds[category_key].get(element, 0.65)
    
    def record_successful_match(self, patient_text: str, guideline_text: str, element: str,
                               category: str, similarity: float, embedding: Optional[np.ndarray] = None):
        """
        Record a successful match for training
        
        Args:
            patient_text: Patient response text
            guideline_text: Guideline term text
            element: OLDCARTS element type
            category: Medical category
            similarity: Final similarity score
            embedding: Optional embedding vector
        """
        if not self.enable_learning:
            return
        
        try:
            features = self.extract_features(
                similarity, patient_text, guideline_text, element, category, embedding
            )
            
            self.training_data['similarity_matches'].append({
                'features': features.tolist(),
                'label': 1.0,  # Successful match
                'element': element,
                'category': category,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 10000 examples to prevent memory issues
            if len(self.training_data['similarity_matches']) > 10000:
                self.training_data['similarity_matches'] = self.training_data['similarity_matches'][-10000:]
            
            # Auto-save periodically
            if len(self.training_data['similarity_matches']) % 100 == 0:
                self._save_training_data()
        except Exception as e:
            print(f"[ML] ⚠️ Error recording successful match: {e}")
    
    def record_failed_match(self, patient_text: str, guideline_text: str, element: str,
                           category: str, similarity: float, embedding: Optional[np.ndarray] = None):
        """
        Record a failed match for negative learning
        
        Args:
            patient_text: Patient response text
            guideline_text: Guideline term text
            element: OLDCARTS element type
            category: Medical category
            similarity: Similarity score that was too low
            embedding: Optional embedding vector
        """
        if not self.enable_learning:
            return
        
        try:
            features = self.extract_features(
                similarity, patient_text, guideline_text, element, category, embedding
            )
            
            self.training_data['failed_matches'].append({
                'features': features.tolist(),
                'label': 0.0,  # Failed match
                'element': element,
                'category': category,
                'timestamp': datetime.now().isoformat()
            })
            
            # Keep only last 5000 examples
            if len(self.training_data['failed_matches']) > 5000:
                self.training_data['failed_matches'] = self.training_data['failed_matches'][-5000:]
        except Exception as e:
            print(f"[ML] ⚠️ Error recording failed match: {e}")
    
    def train_similarity_model(self, epochs: int = 10, batch_size: int = 32):
        """
        Train the similarity adjustment model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if not self.enable_learning or not TORCH_AVAILABLE:
            print("[ML] ⚠️ Training disabled or PyTorch not available")
            return
        
        # Prepare training data
        all_matches = self.training_data['similarity_matches'] + self.training_data['failed_matches']
        
        if len(all_matches) < 100:
            print(f"[ML] ⚠️ Not enough training data ({len(all_matches)} examples). Need at least 100.")
            return
        
        try:
            # Initialize model if not exists
            if self.similarity_model is None:
                # Determine input dimension from first example
                input_dim = len(all_matches[0]['features'])
                self.similarity_model = SemanticMatchingNN(input_dim=input_dim)
                self.optimizer_similarity = optim.Adam(self.similarity_model.parameters(), lr=0.001)
            
            # Prepare data
            features = np.array([m['features'] for m in all_matches])
            labels = np.array([m['label'] for m in all_matches])
            
            # Create dataset and dataloader
            dataset = SemanticMatchDataset(features, labels)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Training loop
            criterion = nn.MSELoss()
            self.similarity_model.train()
            
            print(f"[ML] 🎓 Training similarity model with {len(all_matches)} examples...")
            
            for epoch in range(epochs):
                total_loss = 0.0
                for batch_features, batch_labels in dataloader:
                    self.optimizer_similarity.zero_grad()
                    
                    predictions = self.similarity_model(batch_features)
                    loss = criterion(predictions, batch_labels)
                    
                    loss.backward()
                    self.optimizer_similarity.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                print(f"[ML] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            self.similarity_model.eval()
            self._save_models()
            print("[ML] ✅ Similarity model training complete")
        except Exception as e:
            print(f"[ML] ⚠️ Error training similarity model: {e}")
            import traceback
            traceback.print_exc()
    
    def adjust_threshold(self, element: str, category: str, success_rate: float):
        """
        Adjust threshold based on success rate
        
        Args:
            element: OLDCARTS element type
            category: Medical category
            success_rate: Success rate (0-1) for this element/category
        """
        if not self.enable_learning:
            return
        
        category_key = category or 'default'
        current_threshold = self.adaptive_thresholds[category_key].get(element, 0.65)
        
        # Adjust threshold based on success rate
        # If success rate is low, lower threshold (be more permissive)
        # If success rate is high, can raise threshold (be more strict)
        if success_rate < 0.5:
            # Low success rate: lower threshold
            new_threshold = max(0.45, current_threshold - 0.05)
        elif success_rate > 0.8:
            # High success rate: can raise threshold slightly
            new_threshold = min(0.85, current_threshold + 0.02)
        else:
            # Moderate success rate: keep current
            new_threshold = current_threshold
        
        if new_threshold != current_threshold:
            self.adaptive_thresholds[category_key][element] = new_threshold
            self.stats['threshold_adjustments'] += 1
            self._save_adaptive_thresholds()
            print(f"[ML] 📊 Adjusted threshold for {category_key}/{element}: {current_threshold:.2f} → {new_threshold:.2f}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get ML enhancement statistics"""
        return {
            **self.stats,
            'training_examples': len(self.training_data['similarity_matches']) + len(self.training_data['failed_matches']),
            'learned_patterns': len(self.learned_patterns),
            'adaptive_thresholds': len(self.adaptive_thresholds)
        }
    
    def save_all(self):
        """Save all models and data"""
        self._save_models()
        self._save_training_data()
        self._save_learned_patterns()
        self._save_adaptive_thresholds()

