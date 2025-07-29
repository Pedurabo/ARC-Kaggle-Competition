"""
ARC AI System Metrics Collection
Comprehensive monitoring for performance optimization to 30%
"""

import time
import psutil
import threading
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict, deque
import logging

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary, 
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, multiprocess
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available, using basic metrics")

@dataclass
class ModelMetrics:
    """Metrics for individual model performance"""
    model_name: str
    accuracy: float
    inference_time: float
    confidence: float
    error_rate: float
    throughput: float

class ARCMetricsCollector:
    """Comprehensive metrics collection for ARC AI system"""
    
    def __init__(self, enable_prometheus: bool = True):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.logger = logging.getLogger(__name__)
        
        # Initialize metrics
        self._init_metrics()
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.model_performance = defaultdict(list)
        self.error_counts = defaultdict(int)
        
        # System monitoring
        self.system_metrics = {}
        self.start_time = time.time()
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _init_metrics(self):
        """Initialize Prometheus metrics"""
        if not self.enable_prometheus:
            return
            
        # Counters
        self.request_counter = Counter(
            'arc_requests_total',
            'Total number of requests',
            ['model_type', 'status']
        )
        
        self.prediction_counter = Counter(
            'arc_predictions_total',
            'Total number of predictions',
            ['model_type', 'accuracy_range']
        )
        
        self.error_counter = Counter(
            'arc_errors_total',
            'Total number of errors',
            ['error_type', 'model_type']
        )
        
        # Gauges
        self.accuracy_gauge = Gauge(
            'arc_accuracy_current',
            'Current accuracy percentage',
            ['model_type']
        )
        
        self.confidence_gauge = Gauge(
            'arc_confidence_current',
            'Current confidence level',
            ['model_type']
        )
        
        self.system_memory_gauge = Gauge(
            'arc_system_memory_usage',
            'System memory usage in bytes'
        )
        
        self.system_cpu_gauge = Gauge(
            'arc_system_cpu_usage',
            'System CPU usage percentage'
        )
        
        # Histograms
        self.inference_time_histogram = Histogram(
            'arc_inference_time_seconds',
            'Time spent on inference',
            ['model_type'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        )
        
        self.training_time_histogram = Histogram(
            'arc_training_time_seconds',
            'Time spent on training',
            ['model_type'],
            buckets=[60, 300, 600, 1800, 3600]
        )
        
        # Summaries
        self.accuracy_summary = Summary(
            'arc_accuracy_summary',
            'Accuracy statistics',
            ['model_type']
        )
    
    def _start_background_monitoring(self):
        """Start background system monitoring"""
        def monitor_system():
            while True:
                try:
                    self._collect_system_metrics()
                    time.sleep(30)  # Collect every 30 seconds
                except Exception as e:
                    self.logger.error(f"Error in system monitoring: {e}")
        
        monitor_thread = threading.Thread(target=monitor_system, daemon=True)
        monitor_thread.start()
    
    def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            self.system_metrics = {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used': memory.used,
                'memory_available': memory.available,
                'disk_usage': psutil.disk_usage('/').percent
            }
            
            if self.enable_prometheus:
                self.system_cpu_gauge.set(cpu_percent)
                self.system_memory_gauge.set(memory.used)
                
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
    
    def record_prediction(self, model_type: str, accuracy: float, 
                         inference_time: float, confidence: float):
        """Record prediction metrics"""
        # Store in history
        metrics = ModelMetrics(
            model_name=model_type,
            accuracy=accuracy,
            inference_time=inference_time,
            confidence=confidence,
            error_rate=1.0 - accuracy,
            throughput=1.0 / inference_time if inference_time > 0 else 0
        )
        
        self.performance_history.append(metrics)
        self.model_performance[model_type].append(metrics)
        
        # Update Prometheus metrics
        if self.enable_prometheus:
            self.request_counter.labels(model_type=model_type, status='success').inc()
            
            # Categorize accuracy
            if accuracy >= 0.9:
                acc_range = '90-100'
            elif accuracy >= 0.8:
                acc_range = '80-90'
            elif accuracy >= 0.7:
                acc_range = '70-80'
            elif accuracy >= 0.6:
                acc_range = '60-70'
            elif accuracy >= 0.5:
                acc_range = '50-60'
            else:
                acc_range = '0-50'
            
            self.prediction_counter.labels(
                model_type=model_type, 
                accuracy_range=acc_range
            ).inc()
            
            self.accuracy_gauge.labels(model_type=model_type).set(accuracy * 100)
            self.confidence_gauge.labels(model_type=model_type).set(confidence * 100)
            self.inference_time_histogram.labels(model_type=model_type).observe(inference_time)
            self.accuracy_summary.labels(model_type=model_type).observe(accuracy)
    
    def record_error(self, error_type: str, model_type: str, error_message: str):
        """Record error metrics"""
        self.error_counts[error_type] += 1
        
        if self.enable_prometheus:
            self.error_counter.labels(
                error_type=error_type, 
                model_type=model_type
            ).inc()
            self.request_counter.labels(
                model_type=model_type, 
                status='error'
            ).inc()
        
        self.logger.error(f"Error in {model_type}: {error_type} - {error_message}")
    
    def record_training(self, model_type: str, training_time: float, 
                       final_accuracy: float):
        """Record training metrics"""
        if self.enable_prometheus:
            self.training_time_histogram.labels(model_type=model_type).observe(training_time)
            self.accuracy_gauge.labels(model_type=model_type).set(final_accuracy * 100)
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary"""
        if not self.performance_history:
            return {}
        
        # Calculate overall metrics
        total_predictions = len(self.performance_history)
        avg_accuracy = sum(m.accuracy for m in self.performance_history) / total_predictions
        avg_inference_time = sum(m.inference_time for m in self.performance_history) / total_predictions
        avg_confidence = sum(m.confidence for m in self.performance_history) / total_predictions
        
        # Model-specific metrics
        model_summaries = {}
        for model_type, metrics_list in self.model_performance.items():
            if metrics_list:
                model_avg_accuracy = sum(m.accuracy for m in metrics_list) / len(metrics_list)
                model_summaries[model_type] = {
                    'avg_accuracy': model_avg_accuracy,
                    'total_predictions': len(metrics_list),
                    'recent_accuracy': metrics_list[-1].accuracy if metrics_list else 0
                }
        
        return {
            'overall': {
                'total_predictions': total_predictions,
                'avg_accuracy': avg_accuracy,
                'avg_inference_time': avg_inference_time,
                'avg_confidence': avg_confidence,
                'uptime_seconds': time.time() - self.start_time
            },
            'models': model_summaries,
            'errors': dict(self.error_counts),
            'system': self.system_metrics
        }
    
    def get_accuracy_trend(self, model_type: str = None, window: int = 100) -> List[float]:
        """Get accuracy trend over time"""
        if model_type:
            metrics_list = self.model_performance.get(model_type, [])
        else:
            metrics_list = list(self.performance_history)
        
        if len(metrics_list) < window:
            return [m.accuracy for m in metrics_list]
        
        return [m.accuracy for m in metrics_list[-window:]]
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get recommendations for improving to 30% accuracy"""
        recommendations = []
        summary = self.get_performance_summary()
        
        if not summary:
            return ["No performance data available"]
        
        overall_accuracy = summary['overall']['avg_accuracy']
        
        # Accuracy-based recommendations
        if overall_accuracy < 0.15:
            recommendations.append("Current accuracy below 15% - focus on basic pattern recognition")
        elif overall_accuracy < 0.20:
            recommendations.append("Accuracy 15-20% - implement ensemble methods")
        elif overall_accuracy < 0.25:
            recommendations.append("Accuracy 20-25% - optimize model hyperparameters")
        elif overall_accuracy < 0.30:
            recommendations.append("Accuracy 25-30% - fine-tune breakthrough models")
        
        # Performance-based recommendations
        avg_inference_time = summary['overall']['avg_inference_time']
        if avg_inference_time > 5.0:
            recommendations.append("High inference time - optimize model architecture")
        
        # Error-based recommendations
        if self.error_counts:
            top_error = max(self.error_counts.items(), key=lambda x: x[1])
            recommendations.append(f"High {top_error[0]} errors - improve error handling")
        
        # System-based recommendations
        if self.system_metrics:
            if self.system_metrics.get('memory_percent', 0) > 80:
                recommendations.append("High memory usage - optimize memory management")
            if self.system_metrics.get('cpu_percent', 0) > 90:
                recommendations.append("High CPU usage - consider distributed processing")
        
        return recommendations
    
    def export_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if not self.enable_prometheus:
            return "# Prometheus metrics not available"
        
        return generate_latest()
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        self.performance_history.clear()
        self.model_performance.clear()
        self.error_counts.clear()
        self.start_time = time.time()

# Global metrics collector instance
metrics_collector = ARCMetricsCollector()

def get_metrics_collector() -> ARCMetricsCollector:
    """Get the global metrics collector instance"""
    return metrics_collector 