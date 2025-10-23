#!/usr/bin/env python3
"""
Performance Monitoring System
Track ML accuracy and performance metrics
"""

import json
import os
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import threading
import time

class PerformanceMonitor:
    """
    Performance monitoring system for ML models
    """
    
    def __init__(self, 
                 data_dir: str = "./data/learning",
                 metrics_file: str = "performance_metrics.json",
                 report_interval: int = 3600):  # 1 hour
        """
        Initialize performance monitor
        
        Args:
            data_dir: Directory containing learning data
            metrics_file: File to store performance metrics
            report_interval: Interval for generating reports (seconds)
        """
        self.data_dir = Path(data_dir)
        self.metrics_file = self.data_dir / metrics_file
        self.report_interval = report_interval
        
        # Performance tracking
        self.metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'similarity_scores': [],
            'confidence_scores': [],
            'method_usage': {},
            'organ_system_performance': {},
            'condition_performance': {}
        }
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Start monitoring
        self.start_monitoring()
        
        print(f"[Performance Monitor] 📊 Initialized with report interval: {report_interval}s")
    
    def start_monitoring(self):
        """Start performance monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            print(f"[Performance Monitor] 🔄 Monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print(f"[Performance Monitor] ⏹️ Monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.is_monitoring:
            try:
                # Update metrics from learning data
                self._update_metrics()
                
                # Generate performance report
                self._generate_performance_report()
                
                # Sleep for report interval
                time.sleep(self.report_interval)
                
            except Exception as e:
                print(f"[Performance Monitor] ❌ Monitoring error: {e}")
                time.sleep(60)
    
    def track_prediction(self, 
                        prediction: float, 
                        actual: Optional[float] = None,
                        confidence: str = "medium",
                        method: str = "unknown",
                        condition_name: str = "",
                        organ_system: str = ""):
        """
        Track ML prediction performance
        
        Args:
            prediction: Predicted similarity score
            actual: Actual similarity score (if available)
            confidence: Confidence level (high, medium, low)
            method: Method used (hardcoded_rule, ml_prediction, etc.)
            condition_name: Medical condition name
            organ_system: Organ system (GI, CARDIO, etc.)
        """
        try:
            # Track similarity scores
            self.metrics['similarity_scores'].append({
                'timestamp': datetime.now().isoformat(),
                'prediction': prediction,
                'actual': actual,
                'confidence': confidence,
                'method': method,
                'condition_name': condition_name,
                'organ_system': organ_system
            })
            
            # Track confidence scores
            confidence_value = {'high': 1.0, 'medium': 0.5, 'low': 0.0}.get(confidence, 0.5)
            self.metrics['confidence_scores'].append(confidence_value)
            
            # Track method usage
            if method not in self.metrics['method_usage']:
                self.metrics['method_usage'][method] = 0
            self.metrics['method_usage'][method] += 1
            
            # Track organ system performance
            if organ_system:
                if organ_system not in self.metrics['organ_system_performance']:
                    self.metrics['organ_system_performance'][organ_system] = []
                self.metrics['organ_system_performance'][organ_system].append(prediction)
            
            # Track condition performance
            if condition_name:
                if condition_name not in self.metrics['condition_performance']:
                    self.metrics['condition_performance'][condition_name] = []
                self.metrics['condition_performance'][condition_name].append(prediction)
            
            print(f"[Performance Monitor] 📈 Prediction tracked: {condition_name} ({organ_system}) - {prediction:.3f}")
            
        except Exception as e:
            print(f"[Performance Monitor] ❌ Tracking error: {e}")
    
    def track_accuracy(self, 
                      accuracy: float, 
                      condition_name: str = "",
                      organ_system: str = ""):
        """
        Track accuracy metrics
        
        Args:
            accuracy: Accuracy score (0.0-1.0)
            condition_name: Medical condition name
            organ_system: Organ system
        """
        try:
            self.metrics['accuracy'].append({
                'timestamp': datetime.now().isoformat(),
                'accuracy': accuracy,
                'condition_name': condition_name,
                'organ_system': organ_system
            })
            
            print(f"[Performance Monitor] 📊 Accuracy tracked: {accuracy:.3f} ({condition_name})")
            
        except Exception as e:
            print(f"[Performance Monitor] ❌ Accuracy tracking error: {e}")
    
    def _update_metrics(self):
        """Update metrics from learning data files"""
        try:
            # Load feedback data
            feedback_file = self.data_dir / "feedback.json"
            if feedback_file.exists():
                with open(feedback_file, 'r') as f:
                    feedback_data = json.load(f)
                
                # Process recent feedback (last 24 hours)
                recent_feedback = self._get_recent_data(feedback_data, hours=24)
                
                for feedback in recent_feedback:
                    if feedback.get('accuracy') is not None:
                        self.track_accuracy(
                            accuracy=feedback['accuracy'],
                            condition_name=feedback.get('condition_name', ''),
                            organ_system=feedback.get('organ_system', '')
                        )
            
            # Load predictions data
            predictions_file = self.data_dir / "predictions.json"
            if predictions_file.exists():
                with open(predictions_file, 'r') as f:
                    predictions_data = json.load(f)
                
                # Process recent predictions
                recent_predictions = self._get_recent_data(predictions_data, hours=24)
                
                for prediction in recent_predictions:
                    self.track_prediction(
                        prediction=prediction.get('similarity', 0.0),
                        confidence=prediction.get('confidence', 'medium'),
                        method=prediction.get('method', 'unknown'),
                        condition_name=prediction.get('condition_name', ''),
                        organ_system=prediction.get('organ_system', '')
                    )
            
        except Exception as e:
            print(f"[Performance Monitor] ❌ Metrics update error: {e}")
    
    def _get_recent_data(self, data: List[Dict], hours: int = 24) -> List[Dict]:
        """Get recent data within specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = []
        
        for item in data:
            try:
                item_time = datetime.fromisoformat(item['timestamp'])
                if item_time > cutoff_time:
                    recent_data.append(item)
            except:
                continue
        
        return recent_data
    
    def _generate_performance_report(self):
        """Generate performance report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': self._calculate_summary_metrics(),
                'method_usage': self.metrics['method_usage'],
                'organ_system_performance': self._calculate_organ_system_metrics(),
                'condition_performance': self._calculate_condition_metrics(),
                'trends': self._calculate_trends()
            }
            
            # Save report
            report_file = self.data_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            
            print(f"[Performance Monitor] 📊 Performance report generated: {report_file}")
            
        except Exception as e:
            print(f"[Performance Monitor] ❌ Report generation error: {e}")
    
    def _calculate_summary_metrics(self) -> Dict[str, Any]:
        """Calculate summary performance metrics"""
        try:
            # Calculate accuracy metrics from similarity scores and confidence
            accuracy_scores = [m['accuracy'] for m in self.metrics['accuracy']]
            avg_accuracy = np.mean(accuracy_scores) if accuracy_scores else 0.0
            
            # If no explicit accuracy measurements, calculate from similarity and confidence
            if not accuracy_scores and self.metrics['similarity_scores']:
                # Use similarity scores as proxy for accuracy
                similarity_scores = [s['prediction'] for s in self.metrics['similarity_scores']]
                avg_accuracy = np.mean(similarity_scores) if similarity_scores else 0.0
            
            # Calculate similarity metrics
            similarity_scores = [s['prediction'] for s in self.metrics['similarity_scores']]
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
            
            # Calculate confidence metrics
            avg_confidence = np.mean(self.metrics['confidence_scores']) if self.metrics['confidence_scores'] else 0.0
            
            return {
                'avg_accuracy': avg_accuracy,
                'avg_similarity': avg_similarity,
                'avg_confidence': avg_confidence,
                'total_predictions': len(self.metrics['similarity_scores']),
                'total_accuracy_measurements': len(self.metrics['accuracy'])
            }
            
        except Exception as e:
            print(f"[Performance Monitor] ❌ Summary calculation error: {e}")
            return {}
    
    def _calculate_organ_system_metrics(self) -> Dict[str, Any]:
        """Calculate organ system performance metrics"""
        try:
            organ_metrics = {}
            
            for organ_system, scores in self.metrics['organ_system_performance'].items():
                if scores:
                    organ_metrics[organ_system] = {
                        'avg_similarity': np.mean(scores),
                        'min_similarity': np.min(scores),
                        'max_similarity': np.max(scores),
                        'count': len(scores)
                    }
            
            return organ_metrics
            
        except Exception as e:
            print(f"[Performance Monitor] ❌ Organ system calculation error: {e}")
            return {}
    
    def _calculate_condition_metrics(self) -> Dict[str, Any]:
        """Calculate condition performance metrics"""
        try:
            condition_metrics = {}
            
            for condition, scores in self.metrics['condition_performance'].items():
                if scores:
                    condition_metrics[condition] = {
                        'avg_similarity': np.mean(scores),
                        'min_similarity': np.min(scores),
                        'max_similarity': np.max(scores),
                        'count': len(scores)
                    }
            
            return condition_metrics
            
        except Exception as e:
            print(f"[Performance Monitor] ❌ Condition calculation error: {e}")
            return {}
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends"""
        try:
            trends = {}
            
            # Calculate accuracy trend
            if len(self.metrics['accuracy']) >= 2:
                recent_accuracy = [m['accuracy'] for m in self.metrics['accuracy'][-10:]]
                older_accuracy = [m['accuracy'] for m in self.metrics['accuracy'][:-10]]
                
                if older_accuracy:
                    trends['accuracy_trend'] = np.mean(recent_accuracy) - np.mean(older_accuracy)
                else:
                    trends['accuracy_trend'] = 0.0
            
            # Calculate similarity trend
            if len(self.metrics['similarity_scores']) >= 2:
                recent_similarity = [s['prediction'] for s in self.metrics['similarity_scores'][-10:]]
                older_similarity = [s['prediction'] for s in self.metrics['similarity_scores'][:-10]]
                
                if older_similarity:
                    trends['similarity_trend'] = np.mean(recent_similarity) - np.mean(older_similarity)
                else:
                    trends['similarity_trend'] = 0.0
            
            return trends
            
        except Exception as e:
            print(f"[Performance Monitor] ❌ Trend calculation error: {e}")
            return {}
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get current performance summary"""
        return {
            'is_monitoring': self.is_monitoring,
            'metrics_count': {
                'accuracy': len(self.metrics['accuracy']),
                'similarity_scores': len(self.metrics['similarity_scores']),
                'confidence_scores': len(self.metrics['confidence_scores'])
            },
            'method_usage': self.metrics['method_usage'],
            'organ_system_count': len(self.metrics['organ_system_performance']),
            'condition_count': len(self.metrics['condition_performance'])
        }

# Example usage
if __name__ == "__main__":
    monitor = PerformanceMonitor()
    
    # Test tracking
    monitor.track_prediction(
        prediction=0.8,
        confidence="high",
        method="ml_prediction",
        condition_name="Acute Appendicitis",
        organ_system="GI"
    )
    
    monitor.track_accuracy(
        accuracy=0.85,
        condition_name="Acute Appendicitis",
        organ_system="GI"
    )
    
    # Get summary
    summary = monitor.get_performance_summary()
    print(f"📊 Performance Summary: {summary}")
    
    # Stop monitoring
    monitor.stop_monitoring()
