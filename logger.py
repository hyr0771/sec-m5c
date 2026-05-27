"""
Logging and monitoring system for RNA sequence analysis model.
Provides comprehensive logging, metrics tracking, and monitoring capabilities.
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from collections import defaultdict, deque
import threading
from dataclasses import dataclass, asdict


@dataclass
class MetricEntry:
    """Single metric entry with timestamp"""
    name: str
    value: float
    step: int
    timestamp: float
    stage: Optional[str] = None
    epoch: Optional[int] = None


class MetricsTracker:
    """Thread-safe metrics tracking system"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.metrics = defaultdict(lambda: deque(maxlen=buffer_size))
        self.lock = threading.Lock()
        self.start_time = time.time()
    
    def log_metric(self, name: str, value: float, step: int, 
                   stage: Optional[str] = None, epoch: Optional[int] = None) -> None:
        """Log a metric value"""
        with self.lock:
            entry = MetricEntry(
                name=name,
                value=value,
                step=step,
                timestamp=time.time(),
                stage=stage,
                epoch=epoch
            )
            self.metrics[name].append(entry)
    
    def get_metric_history(self, name: str, last_n: Optional[int] = None) -> List[MetricEntry]:
        """Get metric history"""
        with self.lock:
            history = list(self.metrics[name])
            if last_n:
                return history[-last_n:]
            return history
    
    def get_latest_metric(self, name: str) -> Optional[MetricEntry]:
        """Get latest value for a metric"""
        with self.lock:
            if name in self.metrics and self.metrics[name]:
                return self.metrics[name][-1]
            return None
    
    def get_average_metric(self, name: str, last_n: Optional[int] = None) -> Optional[float]:
        """Get average value for a metric"""
        history = self.get_metric_history(name, last_n)
        if not history:
            return None
        return sum(entry.value for entry in history) / len(history)
    
    def get_all_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get summary of all metrics"""
        with self.lock:
            summary = {}
            for name, entries in self.metrics.items():
                if not entries:
                    continue
                
                values = [entry.value for entry in entries]
                latest = entries[-1]
                
                summary[name] = {
                    'latest_value': latest.value,
                    'latest_step': latest.step,
                    'latest_timestamp': latest.timestamp,
                    'count': len(values),
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'stage': latest.stage,
                    'epoch': latest.epoch
                }
            return summary
    
    def clear_metrics(self) -> None:
        """Clear all metrics"""
        with self.lock:
            self.metrics.clear()


class PerformanceMonitor:
    """Performance monitoring for training stages"""
    
    def __init__(self):
        self.stage_timers = {}
        self.epoch_timers = {}
        self.step_timers = {}
        self.memory_usage = deque(maxlen=100)
        self.gpu_usage = deque(maxlen=100)
    
    def start_stage_timer(self, stage: str) -> None:
        """Start timer for a training stage"""
        self.stage_timers[stage] = {'start': time.time(), 'end': None}
    
    def end_stage_timer(self, stage: str) -> float:
        """End timer for a training stage and return duration"""
        if stage in self.stage_timers:
            self.stage_timers[stage]['end'] = time.time()
            return self.stage_timers[stage]['end'] - self.stage_timers[stage]['start']
        return 0.0
    
    def start_epoch_timer(self, stage: str, epoch: int) -> None:
        """Start timer for an epoch"""
        key = f"{stage}_epoch_{epoch}"
        self.epoch_timers[key] = {'start': time.time(), 'end': None}
    
    def end_epoch_timer(self, stage: str, epoch: int) -> float:
        """End timer for an epoch and return duration"""
        key = f"{stage}_epoch_{epoch}"
        if key in self.epoch_timers:
            self.epoch_timers[key]['end'] = time.time()
            return self.epoch_timers[key]['end'] - self.epoch_timers[key]['start']
        return 0.0
    
    def record_step_time(self, stage: str, step: int, duration: float) -> None:
        """Record step processing time"""
        key = f"{stage}_steps"
        if key not in self.step_timers:
            self.step_timers[key] = deque(maxlen=100)
        self.step_timers[key].append({'step': step, 'duration': duration, 'timestamp': time.time()})
    
    def get_average_step_time(self, stage: str, last_n: int = 50) -> Optional[float]:
        """Get average step time for a stage"""
        key = f"{stage}_steps"
        if key in self.step_timers:
            recent_steps = list(self.step_timers[key])[-last_n:]
            if recent_steps:
                return sum(step['duration'] for step in recent_steps) / len(recent_steps)
        return None
    
    def estimate_remaining_time(self, stage: str, current_step: int, total_steps: int) -> Optional[float]:
        """Estimate remaining time for current stage"""
        avg_step_time = self.get_average_step_time(stage)
        if avg_step_time and total_steps > current_step:
            return avg_step_time * (total_steps - current_step)
        return None


class RNALogger:
    """Main logger class for RNA sequence analysis project"""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.metrics_tracker = MetricsTracker()
        self.performance_monitor = PerformanceMonitor()
        
        # Setup logging
        self.logger = self._setup_logger(log_level)
        
        # Log files
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.performance_file = self.log_dir / "performance.json"
        
        # Runtime info
        self.start_time = time.time()
        self.current_stage = None
        self.current_epoch = None
        self.global_step = 0
    
    def _setup_logger(self, log_level: str) -> logging.Logger:
        """Setup Python logger with file and console handlers"""
        logger = logging.getLogger("dna_analysis")
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # File handler
        log_file = self.log_dir / "training.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def set_stage(self, stage: str) -> None:
        """Set current training stage"""
        if self.current_stage:
            self.performance_monitor.end_stage_timer(self.current_stage)
        
        self.current_stage = stage
        self.performance_monitor.start_stage_timer(stage)
        self.logger.info(f"Starting training stage: {stage}")
    
    def set_epoch(self, epoch: int) -> None:
        """Set current epoch"""
        if self.current_epoch is not None and self.current_stage:
            self.performance_monitor.end_epoch_timer(self.current_stage, self.current_epoch)
        
        self.current_epoch = epoch
        if self.current_stage:
            self.performance_monitor.start_epoch_timer(self.current_stage, epoch)
        self.logger.info(f"Starting epoch {epoch} in stage {self.current_stage}")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        """Log a training metric"""
        if step is None:
            step = self.global_step
        
        self.metrics_tracker.log_metric(
            name=name,
            value=value,
            step=step,
            stage=self.current_stage,
            epoch=self.current_epoch
        )
        
        # Write to metrics file
        self._write_metric_to_file(name, value, step)
        
        # Log to console for important metrics
        if name in ['loss', 'accuracy', 'learning_rate']:
            self.logger.info(f"Step {step}: {name} = {value:.6f}")
    
    def log_step_time(self, duration: float) -> None:
        """Log step processing time"""
        if self.current_stage:
            self.performance_monitor.record_step_time(
                self.current_stage, self.global_step, duration
            )
    
    def increment_step(self) -> None:
        """Increment global step counter"""
        self.global_step += 1
    
    def info(self, message: str) -> None:
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str) -> None:
        """Log debug message"""
        self.logger.debug(message)
    
    def log_model_info(self, model_name: str, num_parameters: int, model_size_mb: float) -> None:
        """Log model information"""
        self.logger.info(f"Model: {model_name}")
        self.logger.info(f"Parameters: {num_parameters:,}")
        self.logger.info(f"Model size: {model_size_mb:.2f} MB")
    
    def log_training_progress(self, current_step: int, total_steps: int) -> None:
        """Log training progress with ETA"""
        progress_pct = (current_step / total_steps) * 100
        
        if self.current_stage:
            eta = self.performance_monitor.estimate_remaining_time(
                self.current_stage, current_step, total_steps
            )
            
            if eta:
                eta_str = f"ETA: {eta/3600:.1f}h" if eta > 3600 else f"ETA: {eta/60:.1f}m"
                self.logger.info(f"Progress: {progress_pct:.1f}% ({current_step}/{total_steps}) - {eta_str}")
            else:
                self.logger.info(f"Progress: {progress_pct:.1f}% ({current_step}/{total_steps})")
    
    def _write_metric_to_file(self, name: str, value: float, step: int) -> None:
        """Write metric to JSONL file"""
        metric_data = {
            'name': name,
            'value': value,
            'step': step,
            'stage': self.current_stage,
            'epoch': self.current_epoch,
            'timestamp': time.time()
        }
        
        try:
            with open(self.metrics_file, 'a') as f:
                f.write(json.dumps(metric_data) + '\n')
        except Exception as e:
            self.logger.error(f"Failed to write metric to file: {e}")
    
    def save_performance_summary(self) -> None:
        """Save performance summary to file"""
        summary = {
            'total_runtime': time.time() - self.start_time,
            'stage_timers': self.performance_monitor.stage_timers,
            'epoch_timers': self.performance_monitor.epoch_timers,
            'metrics_summary': self.metrics_tracker.get_all_metrics_summary(),
            'timestamp': time.time()
        }
        
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save performance summary: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        return self.metrics_tracker.get_all_metrics_summary()
    
    def cleanup(self) -> None:
        """Cleanup and finalize logging"""
        if self.current_stage:
            self.performance_monitor.end_stage_timer(self.current_stage)
        
        if self.current_epoch is not None and self.current_stage:
            self.performance_monitor.end_epoch_timer(self.current_stage, self.current_epoch)
        
        self.save_performance_summary()
        self.logger.info("Training completed. Performance summary saved.")


# Global logger instance
_global_logger = None


def get_logger(log_dir: str = "logs", log_level: str = "INFO") -> RNALogger:
    """Get global logger instance"""
    global _global_logger
    if _global_logger is None:
        _global_logger = RNALogger(log_dir, log_level)
    return _global_logger


def set_logger(logger: RNALogger) -> None:
    """Set global logger instance"""
    global _global_logger
    _global_logger = logger


if __name__ == "__main__":
    # Example usage and testing
    logger = RNALogger("test_logs")
    
    # Test basic logging
    logger.info("Starting test")
    
    # Test stage and epoch tracking
    logger.set_stage("stage1")
    logger.set_epoch(1)
    
    # Test metric logging
    for step in range(10):
        logger.log_metric("loss", 1.0 / (step + 1), step)
        logger.log_metric("accuracy", step * 0.1, step)
        logger.log_step_time(0.5)
        logger.increment_step()
        time.sleep(0.1)  # Simulate processing time
    
    # Test progress logging
    logger.log_training_progress(5, 10)
    
    # Get metrics summary
    summary = logger.get_metrics_summary()
    print("Metrics Summary:")
    for name, stats in summary.items():
        print(f"  {name}: latest={stats['latest_value']:.4f}, mean={stats['mean']:.4f}")
    
    # Cleanup
    logger.cleanup()
    print("Test completed!")
