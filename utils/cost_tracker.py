import time
import os
import json
import psutil
import numpy as np
from dotenv import load_dotenv

load_dotenv()

class CostTracker:
    def __init__(self, method_name):
        self.method_name = method_name
        self.start_time = None
        self.end_time = None
        self.api_calls = 0
        self.api_tokens = 0
        self.memory_usage = []
        self.cpu_usage = []
        
    def start(self):
        """Start tracking time and resource usage"""
        self.start_time = time.time()
        self.memory_usage = []
        self.cpu_usage = []
        self._record_usage()
        return self
        
    def _record_usage(self):
        """Record current memory and CPU usage"""
        process = psutil.Process(os.getpid())
        self.memory_usage.append(process.memory_info().rss / 1024 / 1024)  # MB
        self.cpu_usage.append(psutil.cpu_percent(interval=0.1))
        
    def record_api_call(self, tokens_used=0):
        """Record an API call and tokens used"""
        self.api_calls += 1
        self.api_tokens += tokens_used
        
    def stop(self):
        """Stop tracking and calculate metrics"""
        self.end_time = time.time()
        self._record_usage()
        
    def get_execution_time(self):
        """Get execution time in seconds"""
        if self.start_time is None or self.end_time is None:
            return 0
        return self.end_time - self.start_time
        
    def get_api_cost(self, cost_per_1k_tokens=0.01):
        """Calculate API cost based on tokens used"""
        return (self.api_tokens / 1000) * cost_per_1k_tokens
        
    def get_memory_usage(self):
        """Get average and peak memory usage in MB"""
        if not self.memory_usage:
            return {"avg": 0, "peak": 0}
        return {"avg": np.mean(self.memory_usage), "peak": np.max(self.memory_usage)}
        
    def get_cpu_usage(self):
        """Get average CPU usage percentage"""
        if not self.cpu_usage:
            return {"avg": 0, "peak": 0}
        return {"avg": np.mean(self.cpu_usage), "peak": np.max(self.cpu_usage)}
        
    def get_metrics(self):
        """Get all metrics as a dictionary"""
        return {
            "method": self.method_name,
            "execution_time": self.get_execution_time(),
            "api_calls": self.api_calls,
            "api_tokens": self.api_tokens,
            "api_cost": self.get_api_cost(),
            "memory_usage": self.get_memory_usage(),
            "cpu_usage": self.get_cpu_usage()
        }
        
    def to_json(self):
        """Convert metrics to JSON string"""
        return json.dumps(self.get_metrics(), indent=2)
        
    def save_to_file(self, filepath):
        """Save metrics to a JSON file"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(self.to_json())
