"""
Timing utilities for performance monitoring
"""

import time
from functools import wraps
from loguru import logger
from typing import Callable, Any, Dict

def time_function(operation_name: str):
    """
    Decorator to time function execution
    
    Args:
        operation_name: Name of the operation for logging
    
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(f"⏱ {operation_name} took: {execution_time:.3f}s")
            return result
        return wrapper
    return decorator

def time_operation(operation_name: str, func: Callable, *args, **kwargs) -> tuple:
    """
    Time a function call and return both result and timing
    
    Args:
        operation_name: Name of the operation for logging
        func: Function to call
        *args: Function arguments
        **kwargs: Function keyword arguments
    
    Returns:
        Tuple of (result, execution_time)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    execution_time = time.time() - start_time
    logger.info(f" {operation_name} took: {execution_time:.3f}s")
    return result, execution_time

class TimingContext:
    """
    Context manager for timing operations
    """
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        self.execution_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.execution_time = time.time() - self.start_time
        logger.info(f" {self.operation_name} took: {self.execution_time:.3f}s")

class PipelineTimer:
    """
    Class to track timing across multiple pipeline stages
    """
    
    def __init__(self):
        self.timings: Dict[str, float] = {}
        self.total_start_time = time.time()
    
    def time_stage(self, stage_name: str, func: Callable, *args, **kwargs):
        """
        Time a pipeline stage
        
        Args:
            stage_name: Name of the stage
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            Function result
        """
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        self.timings[stage_name] = execution_time
        logger.info(f" {stage_name} took: {execution_time:.3f}s")
        return result
    
    def get_total_time(self) -> float:
        """Get total pipeline execution time"""
        return time.time() - self.total_start_time
    
    def get_timing_summary(self) -> Dict[str, float]:
        """Get complete timing summary"""
        return {
            **self.timings,
            'total_time': self.get_total_time()
        }
    
    def log_summary(self):
        """Log timing summary"""
        total_time = self.get_total_time()
        logger.info(f"PIPELINE SUMMARY - Total: {total_time:.3f}s")
        for stage, time_taken in self.timings.items():
            percentage = (time_taken / total_time) * 100
            logger.info(f"  - {stage}: {time_taken:.3f}s ({percentage:.1f}%)")
