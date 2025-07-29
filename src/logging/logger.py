"""
ARC AI System Logging
Comprehensive logging with ELK stack integration
"""

import logging
import json
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import traceback

try:
    from pythonjsonlogger import jsonlogger
    JSON_LOGGER_AVAILABLE = True
except ImportError:
    JSON_LOGGER_AVAILABLE = False

class ARCLogFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter for ARC AI system"""
    
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        
        # Add timestamp
        log_record['timestamp'] = datetime.utcnow().isoformat()
        
        # Add service information
        log_record['service'] = 'arc-ai'
        log_record['version'] = '1.0.0'
        
        # Add environment
        log_record['environment'] = os.getenv('ENVIRONMENT', 'development')
        
        # Add log level
        log_record['level'] = record.levelname
        
        # Add additional context
        if hasattr(record, 'model_type'):
            log_record['model_type'] = record.model_type
        if hasattr(record, 'task_id'):
            log_record['task_id'] = record.task_id
        if hasattr(record, 'accuracy'):
            log_record['accuracy'] = record.accuracy
        if hasattr(record, 'inference_time'):
            log_record['inference_time'] = record.inference_time

class ARCLogger:
    """Comprehensive logging system for ARC AI"""
    
    def __init__(self, name: str = 'arc-ai', log_level: str = 'INFO'):
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers()
        
        # Performance tracking
        self.performance_logs = []
        self.error_logs = []
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        if JSON_LOGGER_AVAILABLE:
            console_formatter = ARCLogFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s'
            )
        else:
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / 'arc-ai.log')
        file_handler.setLevel(self.log_level)
        
        if JSON_LOGGER_AVAILABLE:
            file_formatter = ARCLogFormatter(
                '%(timestamp)s %(level)s %(name)s %(message)s'
            )
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.FileHandler(log_dir / 'arc-ai-errors.log')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        self.logger.addHandler(error_handler)
        
        # Performance file handler
        perf_handler = logging.FileHandler(log_dir / 'arc-ai-performance.log')
        perf_handler.setLevel(logging.INFO)
        perf_handler.setFormatter(file_formatter)
        self.logger.addHandler(perf_handler)
    
    def log_prediction(self, model_type: str, task_id: str, accuracy: float, 
                      inference_time: float, confidence: float, prediction: Any):
        """Log prediction results"""
        extra = {
            'model_type': model_type,
            'task_id': task_id,
            'accuracy': accuracy,
            'inference_time': inference_time,
            'confidence': confidence,
            'prediction_type': type(prediction).__name__,
            'log_category': 'prediction'
        }
        
        message = f"Prediction completed - Model: {model_type}, Task: {task_id}, Accuracy: {accuracy:.3f}, Time: {inference_time:.3f}s"
        
        if accuracy >= 0.8:
            self.logger.info(message, extra=extra)
        elif accuracy >= 0.6:
            self.logger.warning(message, extra=extra)
        else:
            self.logger.error(message, extra=extra)
        
        # Store for analysis
        self.performance_logs.append({
            'timestamp': datetime.utcnow().isoformat(),
            'model_type': model_type,
            'task_id': task_id,
            'accuracy': accuracy,
            'inference_time': inference_time,
            'confidence': confidence
        })
    
    def log_training(self, model_type: str, epoch: int, loss: float, 
                    accuracy: float, training_time: float):
        """Log training progress"""
        extra = {
            'model_type': model_type,
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
            'training_time': training_time,
            'log_category': 'training'
        }
        
        message = f"Training progress - Model: {model_type}, Epoch: {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.3f}"
        self.logger.info(message, extra=extra)
    
    def log_error(self, error_type: str, model_type: str, error_message: str, 
                  task_id: str = None, stack_trace: str = None):
        """Log errors with context"""
        extra = {
            'error_type': error_type,
            'model_type': model_type,
            'task_id': task_id,
            'stack_trace': stack_trace or traceback.format_exc(),
            'log_category': 'error'
        }
        
        message = f"Error occurred - Type: {error_type}, Model: {model_type}, Message: {error_message}"
        self.logger.error(message, extra=extra)
        
        # Store for analysis
        self.error_logs.append({
            'timestamp': datetime.utcnow().isoformat(),
            'error_type': error_type,
            'model_type': model_type,
            'task_id': task_id,
            'error_message': error_message
        })
    
    def log_system(self, component: str, message: str, metrics: Dict[str, Any] = None):
        """Log system-level information"""
        extra = {
            'component': component,
            'metrics': metrics or {},
            'log_category': 'system'
        }
        
        self.logger.info(f"System: {component} - {message}", extra=extra)
    
    def log_security(self, event_type: str, user: str = None, 
                    resource: str = None, details: Dict[str, Any] = None):
        """Log security events"""
        extra = {
            'event_type': event_type,
            'user': user,
            'resource': resource,
            'details': details or {},
            'log_category': 'security'
        }
        
        message = f"Security event - Type: {event_type}"
        if user:
            message += f", User: {user}"
        if resource:
            message += f", Resource: {resource}"
        
        self.logger.warning(message, extra=extra)
    
    def log_performance_optimization(self, optimization_type: str, 
                                   before_metrics: Dict[str, Any], 
                                   after_metrics: Dict[str, Any]):
        """Log performance optimization results"""
        improvement = {}
        for key in before_metrics:
            if key in after_metrics:
                if isinstance(before_metrics[key], (int, float)) and isinstance(after_metrics[key], (int, float)):
                    improvement[key] = after_metrics[key] - before_metrics[key]
        
        extra = {
            'optimization_type': optimization_type,
            'before_metrics': before_metrics,
            'after_metrics': after_metrics,
            'improvement': improvement,
            'log_category': 'optimization'
        }
        
        message = f"Performance optimization - Type: {optimization_type}, Improvement: {improvement}"
        self.logger.info(message, extra=extra)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance logs"""
        if not self.performance_logs:
            return {}
        
        # Calculate statistics
        accuracies = [log['accuracy'] for log in self.performance_logs]
        inference_times = [log['inference_time'] for log in self.performance_logs]
        
        return {
            'total_predictions': len(self.performance_logs),
            'avg_accuracy': sum(accuracies) / len(accuracies),
            'max_accuracy': max(accuracies),
            'min_accuracy': min(accuracies),
            'avg_inference_time': sum(inference_times) / len(inference_times),
            'model_performance': self._get_model_performance()
        }
    
    def _get_model_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance by model type"""
        model_stats = {}
        
        for log in self.performance_logs:
            model_type = log['model_type']
            if model_type not in model_stats:
                model_stats[model_type] = {
                    'count': 0,
                    'total_accuracy': 0,
                    'total_time': 0
                }
            
            model_stats[model_type]['count'] += 1
            model_stats[model_type]['total_accuracy'] += log['accuracy']
            model_stats[model_type]['total_time'] += log['inference_time']
        
        # Calculate averages
        for model_type, stats in model_stats.items():
            count = stats['count']
            stats['avg_accuracy'] = stats['total_accuracy'] / count
            stats['avg_inference_time'] = stats['total_time'] / count
            del stats['total_accuracy']
            del stats['total_time']
        
        return model_stats
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of error logs"""
        if not self.error_logs:
            return {}
        
        error_counts = {}
        model_error_counts = {}
        
        for log in self.error_logs:
            error_type = log['error_type']
            model_type = log['model_type']
            
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            model_error_counts[model_type] = model_error_counts.get(model_type, 0) + 1
        
        return {
            'total_errors': len(self.error_logs),
            'error_types': error_counts,
            'model_errors': model_error_counts
        }
    
    def export_logs(self, log_type: str = 'all') -> str:
        """Export logs in JSON format"""
        if log_type == 'performance':
            return json.dumps(self.performance_logs, indent=2)
        elif log_type == 'errors':
            return json.dumps(self.error_logs, indent=2)
        else:
            return json.dumps({
                'performance': self.performance_logs,
                'errors': self.error_logs
            }, indent=2)

# Global logger instance
arc_logger = ARCLogger()

def get_logger() -> ARCLogger:
    """Get the global logger instance"""
    return arc_logger 