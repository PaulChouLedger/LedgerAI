#!/usr/bin/env python3
"""
Performance Dashboard - Monitor learning system performance
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import numpy as np

class PerformanceDashboard:
    """
    Performance dashboard for monitoring learning system
    """
    
    def __init__(self, data_dir: str = "./data/learning"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Data files
        self.feedback_file = self.data_dir / "feedback.json"
        self.predictions_file = self.data_dir / "predictions.json"
        self.performance_file = self.data_dir / "performance.json"
        self.user_feedback_file = self.data_dir / "user_feedback.json"
        
        print(f"[Performance Dashboard] 📊 Initialized - Data directory: {self.data_dir}")
    
    def get_performance_overview(self) -> Dict[str, Any]:
        """Get comprehensive performance overview"""
        overview = {
            'timestamp': datetime.now().isoformat(),
            'data_availability': self._check_data_availability(),
            'learning_metrics': self._get_learning_metrics(),
            'performance_trends': self._get_performance_trends(),
            'user_satisfaction': self._get_user_satisfaction(),
            'system_health': self._get_system_health()
        }
        
        return overview
    
    def _check_data_availability(self) -> Dict[str, Any]:
        """Check availability of learning data files"""
        availability = {}
        
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
                    availability[file_name] = {
                        'available': True,
                        'count': len(data),
                        'last_updated': self._get_file_modified_time(file_path)
                    }
                except:
                    availability[file_name] = {
                        'available': True,
                        'count': 0,
                        'error': 'Failed to parse JSON'
                    }
            else:
                availability[file_name] = {
                    'available': False,
                    'count': 0
                }
        
        return availability
    
    def _get_file_modified_time(self, file_path: Path) -> str:
        """Get file modification time"""
        try:
            timestamp = file_path.stat().st_mtime
            return datetime.fromtimestamp(timestamp).isoformat()
        except:
            return "Unknown"
    
    def _get_learning_metrics(self) -> Dict[str, Any]:
        """Get learning metrics"""
        metrics = {
            'total_predictions': 0,
            'total_feedback': 0,
            'avg_accuracy': 0.0,
            'avg_similarity': 0.0,
            'method_distribution': {},
            'organ_system_distribution': {},
            'condition_distribution': {}
        }
        
        try:
            # Load predictions data
            if self.predictions_file.exists():
                with open(self.predictions_file, 'r') as f:
                    predictions = json.load(f)
                
                metrics['total_predictions'] = len(predictions)
                
                similarity_scores = []
                for prediction in predictions:
                    if 'similarity' in prediction:
                        similarity_scores.append(prediction['similarity'])
                    
                    # Method distribution
                    method = prediction.get('method', 'unknown')
                    metrics['method_distribution'][method] = metrics['method_distribution'].get(method, 0) + 1
                    
                    # Organ system distribution
                    organ_system = prediction.get('organ_system', 'unknown')
                    metrics['organ_system_distribution'][organ_system] = metrics['organ_system_distribution'].get(organ_system, 0) + 1
                    
                    # Condition distribution
                    condition = prediction.get('condition_name', 'unknown')
                    metrics['condition_distribution'][condition] = metrics['condition_distribution'].get(condition, 0) + 1
                
                if similarity_scores:
                    metrics['avg_similarity'] = np.mean(similarity_scores)
            
            # Load feedback data
            if self.feedback_file.exists():
                with open(self.feedback_file, 'r') as f:
                    feedback = json.load(f)
                
                metrics['total_feedback'] = len(feedback)
                
                accuracy_scores = []
                for fb in feedback:
                    if 'accuracy' in fb and fb['accuracy'] is not None:
                        accuracy_scores.append(fb['accuracy'])
                
                if accuracy_scores:
                    metrics['avg_accuracy'] = np.mean(accuracy_scores)
            
        except Exception as e:
            metrics['error'] = str(e)
        
        return metrics
    
    def _get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trends over time"""
        trends = {
            'accuracy_trend': 0.0,
            'similarity_trend': 0.0,
            'recent_performance': {},
            'performance_by_organ_system': {},
            'performance_by_condition': {}
        }
        
        try:
            # Load performance data
            if self.performance_file.exists():
                with open(self.performance_file, 'r') as f:
                    performance_data = json.load(f)
                
                # Calculate trends (last 10 vs previous 10)
                if len(performance_data) >= 20:
                    recent_data = performance_data[-10:]
                    older_data = performance_data[-20:-10]
                    
                    recent_accuracy = [p['value'] for p in recent_data if p.get('metric_name') == 'accuracy']
                    older_accuracy = [p['value'] for p in older_data if p.get('metric_name') == 'accuracy']
                    
                    if recent_accuracy and older_accuracy:
                        trends['accuracy_trend'] = np.mean(recent_accuracy) - np.mean(older_accuracy)
                
                # Recent performance (last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                recent_performance = []
                
                for perf in performance_data:
                    try:
                        perf_time = datetime.fromisoformat(perf['timestamp'])
                        if perf_time > cutoff_time:
                            recent_performance.append(perf)
                    except:
                        continue
                
                if recent_performance:
                    trends['recent_performance'] = {
                        'count': len(recent_performance),
                        'avg_value': np.mean([p['value'] for p in recent_performance if 'value' in p])
                    }
                
                # Performance by organ system
                for perf in performance_data:
                    organ_system = perf.get('organ_system', 'unknown')
                    if organ_system not in trends['performance_by_organ_system']:
                        trends['performance_by_organ_system'][organ_system] = []
                    if 'value' in perf:
                        trends['performance_by_organ_system'][organ_system].append(perf['value'])
                
                # Calculate averages for organ systems
                for organ_system, values in trends['performance_by_organ_system'].items():
                    if values:
                        trends['performance_by_organ_system'][organ_system] = {
                            'count': len(values),
                            'avg_value': np.mean(values)
                        }
                
        except Exception as e:
            trends['error'] = str(e)
        
        return trends
    
    def _get_user_satisfaction(self) -> Dict[str, Any]:
        """Get user satisfaction metrics"""
        satisfaction = {
            'total_ratings': 0,
            'average_rating': 0.0,
            'rating_distribution': {},
            'feedback_types': {},
            'recent_satisfaction': {}
        }
        
        try:
            if self.user_feedback_file.exists():
                with open(self.user_feedback_file, 'r') as f:
                    user_feedback = json.load(f)
                
                ratings = []
                for feedback in user_feedback:
                    if 'user_rating' in feedback:
                        rating = feedback['user_rating']
                        ratings.append(rating)
                        satisfaction['rating_distribution'][rating] = satisfaction['rating_distribution'].get(rating, 0) + 1
                    
                    # Feedback types
                    feedback_type = feedback.get('feedback_type', 'unknown')
                    satisfaction['feedback_types'][feedback_type] = satisfaction['feedback_types'].get(feedback_type, 0) + 1
                
                satisfaction['total_ratings'] = len(ratings)
                if ratings:
                    satisfaction['average_rating'] = np.mean(ratings)
                
                # Recent satisfaction (last 24 hours)
                cutoff_time = datetime.now() - timedelta(hours=24)
                recent_ratings = []
                
                for feedback in user_feedback:
                    try:
                        feedback_time = datetime.fromisoformat(feedback['timestamp'])
                        if feedback_time > cutoff_time and 'user_rating' in feedback:
                            recent_ratings.append(feedback['user_rating'])
                    except:
                        continue
                
                if recent_ratings:
                    satisfaction['recent_satisfaction'] = {
                        'count': len(recent_ratings),
                        'average_rating': np.mean(recent_ratings)
                    }
                
        except Exception as e:
            satisfaction['error'] = str(e)
        
        return satisfaction
    
    def _get_system_health(self) -> Dict[str, Any]:
        """Get system health indicators"""
        health = {
            'data_freshness': {},
            'learning_activity': {},
            'performance_indicators': {},
            'overall_health': 'unknown'
        }
        
        try:
            # Check data freshness
            for file_name, file_path in [
                ('feedback', self.feedback_file),
                ('predictions', self.predictions_file),
                ('performance', self.performance_file),
                ('user_feedback', self.user_feedback_file)
            ]:
                if file_path.exists():
                    try:
                        timestamp = file_path.stat().st_mtime
                        last_modified = datetime.fromtimestamp(timestamp)
                        hours_ago = (datetime.now() - last_modified).total_seconds() / 3600
                        
                        health['data_freshness'][file_name] = {
                            'last_modified': last_modified.isoformat(),
                            'hours_ago': round(hours_ago, 2),
                            'fresh': hours_ago < 24  # Fresh if less than 24 hours
                        }
                    except:
                        health['data_freshness'][file_name] = {
                            'error': 'Failed to get modification time'
                        }
            
            # Check learning activity
            recent_activity = self._get_recent_activity(24)
            health['learning_activity'] = recent_activity
            
            # Performance indicators
            health['performance_indicators'] = {
                'has_recent_predictions': recent_activity.get('recent_predictions', 0) > 0,
                'has_recent_feedback': recent_activity.get('recent_feedback', 0) > 0,
                'has_recent_user_feedback': recent_activity.get('recent_user_feedback', 0) > 0
            }
            
            # Overall health assessment
            health_score = 0
            if health['performance_indicators']['has_recent_predictions']:
                health_score += 1
            if health['performance_indicators']['has_recent_feedback']:
                health_score += 1
            if health['performance_indicators']['has_recent_user_feedback']:
                health_score += 1
            
            if health_score >= 2:
                health['overall_health'] = 'healthy'
            elif health_score == 1:
                health['overall_health'] = 'moderate'
            else:
                health['overall_health'] = 'needs_attention'
            
        except Exception as e:
            health['error'] = str(e)
        
        return health
    
    def _get_recent_activity(self, hours: int = 24) -> Dict[str, int]:
        """Get recent activity within specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        activity = {
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
                        pred_time = datetime.fromisoformat(prediction['timestamp'])
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
                        fb_time = datetime.fromisoformat(fb['timestamp'])
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
                        perf_time = datetime.fromisoformat(perf['timestamp'])
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
                        uf_time = datetime.fromisoformat(uf['timestamp'])
                        if uf_time > cutoff_time:
                            activity['recent_user_feedback'] += 1
                    except:
                        continue
            
        except Exception as e:
            activity['error'] = str(e)
        
        return activity
    
    def print_dashboard(self):
        """Print formatted performance dashboard"""
        overview = self.get_performance_overview()
        
        print("📊 PERFORMANCE DASHBOARD")
        print("=" * 50)
        print(f"Timestamp: {overview['timestamp']}")
        
        print("\n📁 DATA AVAILABILITY")
        print("-" * 30)
        for file_name, status in overview['data_availability'].items():
            if status['available']:
                print(f"✅ {file_name}: {status['count']} records")
            else:
                print(f"❌ {file_name}: Not available")
        
        print("\n📈 LEARNING METRICS")
        print("-" * 30)
        metrics = overview['learning_metrics']
        print(f"Total Predictions: {metrics['total_predictions']}")
        print(f"Total Feedback: {metrics['total_feedback']}")
        print(f"Average Accuracy: {metrics['avg_accuracy']:.3f}")
        print(f"Average Similarity: {metrics['avg_similarity']:.3f}")
        
        print("\n🎯 PERFORMANCE TRENDS")
        print("-" * 30)
        trends = overview['performance_trends']
        print(f"Accuracy Trend: {trends['accuracy_trend']:+.3f}")
        print(f"Recent Performance: {trends['recent_performance']}")
        
        print("\n😊 USER SATISFACTION")
        print("-" * 30)
        satisfaction = overview['user_satisfaction']
        print(f"Total Ratings: {satisfaction['total_ratings']}")
        print(f"Average Rating: {satisfaction['average_rating']:.1f}/5")
        print(f"Rating Distribution: {satisfaction['rating_distribution']}")
        
        print("\n🏥 SYSTEM HEALTH")
        print("-" * 30)
        health = overview['system_health']
        print(f"Overall Health: {health['overall_health'].upper()}")
        print(f"Recent Activity: {health['learning_activity']}")
        print(f"Performance Indicators: {health['performance_indicators']}")

# Example usage
if __name__ == "__main__":
    dashboard = PerformanceDashboard()
    dashboard.print_dashboard()
