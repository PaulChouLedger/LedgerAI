#!/usr/bin/env python3
"""
Learning Tracker - Monitor and track learning system performance
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

class LearningTracker:
    """
    Track and monitor learning system performance
    """
    
    def __init__(self, data_dir: str = "./data/learning"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Learning data files
        self.feedback_file = self.data_dir / "feedback.json"
        self.predictions_file = self.data_dir / "predictions.json"
        self.performance_file = self.data_dir / "performance.json"
        self.user_feedback_file = self.data_dir / "user_feedback.json"
        
        print(f"[Learning Tracker] 📊 Initialized - Data directory: {self.data_dir}")
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Get comprehensive learning summary"""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'data_files': {},
            'learning_metrics': {},
            'performance_metrics': {},
            'user_feedback_metrics': {}
        }
        
        # Check data files
        for file_name, file_path in [
            ('feedback', self.feedback_file),
            ('predictions', self.predictions_file),
            ('performance', self.performance_file),
            ('user_feedback', self.user_feedback_file)
        ]:
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    summary['data_files'][file_name] = {
                        'exists': True,
                        'count': len(data),
                        'last_updated': self._get_file_modified_time(file_path)
                    }
                except:
                    summary['data_files'][file_name] = {
                        'exists': True,
                        'count': 0,
                        'error': 'Failed to parse JSON'
                    }
            else:
                summary['data_files'][file_name] = {
                    'exists': False,
                    'count': 0
                }
        
        # Calculate learning metrics
        summary['learning_metrics'] = self._calculate_learning_metrics()
        
        # Calculate performance metrics
        summary['performance_metrics'] = self._calculate_performance_metrics()
        
        # Calculate user feedback metrics
        summary['user_feedback_metrics'] = self._calculate_user_feedback_metrics()
        
        return summary
    
    def _get_file_modified_time(self, file_path: Path) -> str:
        """Get file modification time"""
        try:
            timestamp = file_path.stat().st_mtime
            return datetime.fromtimestamp(timestamp).isoformat()
        except:
            return "Unknown"
    
    def _calculate_learning_metrics(self) -> Dict[str, Any]:
        """Calculate learning metrics from data files"""
        metrics = {
            'total_predictions': 0,
            'total_feedback': 0,
            'accuracy_scores': [],
            'similarity_scores': [],
            'methods_used': {},
            'organ_systems': {},
            'conditions': {}
        }
        
        try:
            # Load predictions data
            if self.predictions_file.exists():
                with open(self.predictions_file, 'r') as f:
                    predictions = json.load(f)
                
                metrics['total_predictions'] = len(predictions)
                
                for prediction in predictions:
                    # Similarity scores
                    if 'similarity' in prediction:
                        metrics['similarity_scores'].append(prediction['similarity'])
                    
                    # Methods used
                    method = prediction.get('method', 'unknown')
                    metrics['methods_used'][method] = metrics['methods_used'].get(method, 0) + 1
                    
                    # Organ systems
                    organ_system = prediction.get('organ_system', 'unknown')
                    metrics['organ_systems'][organ_system] = metrics['organ_systems'].get(organ_system, 0) + 1
                    
                    # Conditions
                    condition = prediction.get('condition_name', 'unknown')
                    metrics['conditions'][condition] = metrics['conditions'].get(condition, 0) + 1
            
            # Load feedback data
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r') as f:
                    feedback = json.load(f)
                
                metrics['total_feedback'] = len(feedback)
                
                for fb in feedback:
                    if 'accuracy' in fb and fb['accuracy'] is not None:
                        metrics['accuracy_scores'].append(fb['accuracy'])
            
            # Calculate averages
            if metrics['accuracy_scores']:
                metrics['avg_accuracy'] = sum(metrics['accuracy_scores']) / len(metrics['accuracy_scores'])
            else:
                metrics['avg_accuracy'] = 0.0
            
            if metrics['similarity_scores']:
                metrics['avg_similarity'] = sum(metrics['similarity_scores']) / len(metrics['similarity_scores'])
            else:
                metrics['avg_similarity'] = 0.0
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics"""
        metrics = {
            'total_performance_measurements': 0,
            'performance_scores': [],
            'metric_types': {},
            'organ_system_performance': {},
            'condition_performance': {}
        }
        
        try:
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    performance_data = json.load(f)
                
                metrics['total_performance_measurements'] = len(performance_data)
                
                for perf in performance_data:
                    # Performance scores
                    if 'value' in perf:
                        metrics['performance_scores'].append(perf['value'])
                    
                    # Metric types
                    metric_type = perf.get('metric_name', 'unknown')
                    metrics['metric_types'][metric_type] = metrics['metric_types'].get(metric_type, 0) + 1
                    
                    # Organ system performance
                    organ_system = perf.get('organ_system', 'unknown')
                    if organ_system not in metrics['organ_system_performance']:
                        metrics['organ_system_performance'][organ_system] = []
                    if 'value' in perf:
                        metrics['organ_system_performance'][organ_system].append(perf['value'])
                    
                    # Condition performance
                    condition = perf.get('condition_name', 'unknown')
                    if condition not in metrics['condition_performance']:
                        metrics['condition_performance'][condition] = []
                    if 'value' in perf:
                        metrics['condition_performance'][condition].append(perf['value'])
                
                # Calculate averages
                if metrics['performance_scores']:
                    metrics['avg_performance'] = sum(metrics['performance_scores']) / len(metrics['performance_scores'])
                else:
                    metrics['avg_performance'] = 0.0
                
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def _calculate_user_feedback_metrics(self) -> Dict[str, Any]:
        """Calculate user feedback metrics"""
        metrics = {
            'total_user_feedback': 0,
            'ratings': [],
            'feedback_types': {},
            'average_rating': 0.0,
            'comments': []
        }
        
        try:
            if self.user_feedback_file.exists():
                with open(self.user_feedback_file, 'r') as f:
                    user_feedback = json.load(f)
                
                metrics['total_user_feedback'] = len(user_feedback)
                
                for feedback in user_feedback:
                    # Ratings
                    if 'user_rating' in feedback:
                        metrics['ratings'].append(feedback['user_rating'])
                    
                    # Feedback types
                    feedback_type = feedback.get('feedback_type', 'unknown')
                    metrics['feedback_types'][feedback_type] = metrics['feedback_types'].get(feedback_type, 0) + 1
                    
                    # Comments
                    if 'user_comment' in feedback and feedback['user_comment']:
                        metrics['comments'].append(feedback['user_comment'])
                
                # Calculate average rating
                if metrics['ratings']:
                    metrics['average_rating'] = sum(metrics['ratings']) / len(metrics['ratings'])
                else:
                    metrics['average_rating'] = 0.0
                
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def get_recent_activity(self, hours: int = 24) -> Dict[str, Any]:
        """Get recent activity within specified hours"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        activity = {
            'timeframe_hours': hours,
            'recent_predictions': 0,
            'recent_feedback': 0,
            'recent_performance': 0,
            'recent_user_feedback': 0
        }
        
        try:
            # Check recent predictions
            if self.predictions_file.exists():
                with open(self.predictions_file, 'r') as f:
                    predictions = json.load(f)
                
                for prediction in predictions:
                    try:
                        pred_time = datetime.fromisoformat(prediction['timestamp']).timestamp()
                        if pred_time > cutoff_time:
                            activity['recent_predictions'] += 1
                    except:
                        continue
            
            # Check recent feedback
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r') as f:
                    feedback = json.load(f)
                
                for fb in feedback:
                    try:
                        fb_time = datetime.fromisoformat(fb['timestamp']).timestamp()
                        if fb_time > cutoff_time:
                            activity['recent_feedback'] += 1
                    except:
                        continue
            
            # Check recent performance
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    performance = json.load(f)
                
                for perf in performance:
                    try:
                        perf_time = datetime.fromisoformat(perf['timestamp']).timestamp()
                        if perf_time > cutoff_time:
                            activity['recent_performance'] += 1
                    except:
                        continue
            
            # Check recent user feedback
            if self.user_feedback_file.exists():
                with open(self.user_feedback_file, 'r') as f:
                    user_feedback = json.load(f)
                
                for uf in user_feedback:
                    try:
                        uf_time = datetime.fromisoformat(uf['timestamp']).timestamp()
                        if uf_time > cutoff_time:
                            activity['recent_user_feedback'] += 1
                    except:
                        continue
            
        except Exception as e:
            activity['error'] = str(e)
        
        return activity
    
    def export_learning_data(self, output_file: str = "learning_export.json") -> Path:
        """Export all learning data to a single file"""
        try:
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'summary': self.get_learning_summary(),
                'recent_activity': self.get_recent_activity(24),
                'data_files': {}
            }
            
            # Export individual data files
            for file_name, file_path in [
                ('feedback', self.feedback_file),
                ('predictions', self.predictions_file),
                ('performance', self.performance_file),
                ('user_feedback', self.user_feedback_file)
            ]:
                if file_path.exists():
                    with open(file_path, 'r') as f:
                        export_data['data_files'][file_name] = json.load(f)
                else:
                    export_data['data_files'][file_name] = []
            
            # Save export file
            export_path = self.data_dir / output_file
            with open(export_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            print(f"[Learning Tracker] 📤 Learning data exported to {export_path}")
            return export_path
            
        except Exception as e:
            print(f"[Learning Tracker] ❌ Export error: {e}")
            return None

# Example usage
if __name__ == "__main__":
    tracker = LearningTracker()
    
    # Get learning summary
    summary = tracker.get_learning_summary()
    print("📊 Learning Summary:")
    print(f"  Data Files: {summary['data_files']}")
    print(f"  Learning Metrics: {summary['learning_metrics']}")
    print(f"  Performance Metrics: {summary['performance_metrics']}")
    print(f"  User Feedback Metrics: {summary['user_feedback_metrics']}")
    
    # Get recent activity
    activity = tracker.get_recent_activity(24)
    print(f"\n🕐 Recent Activity (24h): {activity}")
    
    # Export data
    export_path = tracker.export_learning_data()
    if export_path:
        print(f"📤 Data exported to: {export_path}")
