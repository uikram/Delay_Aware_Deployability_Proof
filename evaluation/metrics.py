"""
Comprehensive metrics computation and storage - FIXED VERSION.
"""

import json
import time
import torch
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime


class MetricsTracker:
    """Unified metrics tracking for all multimodal models."""
        
    def __init__(self, model_name: str, results_dir: Path):
        self.model_name = model_name.upper()
        
        if isinstance(results_dir, str):
            results_dir = Path(results_dir)
        
        self.results_dir = results_dir / self.model_name
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'model_name': self.model_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': {},
            'memory': {},
            'latency': {},
            'training_time': {},
            'evaluation_results': {},
            'training_history': {}
        }
        
        self.training_history = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'train_accuracy': [],
            'val_accuracy': []
        }
        
        # Timing trackers
        self.train_start_time = None
        self.train_end_time = None
        self.eval_start_time = None
        self.eval_end_time = None
        self.inference_start_time = None
        self.inference_end_time = None

    def track_parameters(self, model):
        """Track model parameters with correct counting for all model types."""
        # For LoRA models, get the underlying PEFT model
        if hasattr(model, 'model') and hasattr(model.model, 'print_trainable_parameters'):
            base_model = model.model
            total_params = sum(p.numel() for p in base_model.parameters())
            trainable_params = sum(p.numel() for p in base_model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
            
        elif hasattr(model, 'count_parameters'):
            param_counts = model.count_parameters()
            total_params = param_counts['total']
            trainable_params = param_counts['trainable']
            frozen_params = total_params - trainable_params
            
        else:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
        
        self.metrics['parameters'] = {
            'total_parameters': int(total_params),
            'trainable_parameters': int(trainable_params),
            'frozen_parameters': int(frozen_params),
            'trainable_percentage': round((trainable_params / total_params) * 100, 4) if total_params > 0 else 0.0
        }
        
        print(f"\n📊 Parameter Counts for {self.model_name}:")
        print(f"   Total: {total_params:,}")
        print(f"   Trainable: {trainable_params:,} ({self.metrics['parameters']['trainable_percentage']:.2f}%)")
        print(f"   Frozen: {frozen_params:,}")
        
    def track_epoch_metrics(self, epoch: int, train_loss: float = None, 
                            val_loss: float = None, train_acc: float = None, 
                            val_acc: float = None):
        """Track per-epoch training metrics."""
        self.training_history['epochs'].append(int(epoch))
        
        if train_loss is not None:
            self.training_history['train_loss'].append(float(train_loss))
        if val_loss is not None:
            self.training_history['val_loss'].append(float(val_loss))
        if train_acc is not None:
            self.training_history['train_accuracy'].append(float(train_acc))
        if val_acc is not None:
            self.training_history['val_accuracy'].append(float(val_acc))
        
        self.metrics['training_history'] = self.training_history
 
    def start_training_timer(self):
        """Start training timer."""
        self.train_start_time = time.time()
        
    def end_training_timer(self):
        """End training timer and record."""
        if self.train_start_time is None:
            return
            
        self.train_end_time = time.time()
        total_seconds = self.train_end_time - self.train_start_time
        
        self.metrics['training_time'] = {
            'total_seconds': round(total_seconds, 2),
            'total_minutes': round(total_seconds / 60, 2),
            'total_hours': round(total_seconds / 3600, 4)
        }
    
    def start_evaluation_timer(self):
        """Start evaluation timer."""
        self.eval_start_time = time.time()
        
    def end_evaluation_timer(self):
        """End evaluation timer."""
        if self.eval_start_time is None:
            return
        self.eval_end_time = time.time()
    
    def start_inference_timer(self):
        """Start inference timer."""
        self.inference_start_time = time.time()
        
    def end_inference_timer(self):
        """End inference timer and record."""
        if hasattr(self, 'inference_start_time') and self.inference_start_time is not None:
            self.inference_end_time = time.time()
            total_time = self.inference_end_time - self.inference_start_time
            
            self.metrics['inference_time'] = {
                'total_seconds': round(total_time, 2),
                'total_minutes': round(total_time / 60, 2),
                'total_hours': round(total_time / 3600, 4)
            }
        
    def track_gpu_memory(self, stage: str = 'training'):
        """Track GPU memory usage."""
        if torch.cuda.is_available():
            current_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            reserved_memory_mb = torch.cuda.memory_reserved() / (1024 ** 2)
            
            self.metrics['memory'][f'{stage}_current_gpu_mb'] = round(current_memory_mb, 2)
            self.metrics['memory'][f'{stage}_peak_gpu_mb'] = round(peak_memory_mb, 2)
            self.metrics['memory'][f'{stage}_reserved_gpu_mb'] = round(reserved_memory_mb, 2)
            
            print(f"\n💾 GPU Memory ({stage}):")
            print(f"   Current: {current_memory_mb:.2f} MB")
            print(f"   Peak: {peak_memory_mb:.2f} MB")
        else:
            self.metrics['memory'][f'{stage}_current_gpu_mb'] = 'N/A'
            self.metrics['memory'][f'{stage}_peak_gpu_mb'] = 'N/A'
            
    def track_cpu_memory(self, stage: str = 'current'):
        """Track CPU memory usage."""
        process = psutil.Process()
        cpu_memory_mb = process.memory_info().rss / (1024 ** 2)
        self.metrics['memory'][f'{stage}_cpu_memory_mb'] = round(cpu_memory_mb, 2)
        
    def track_performance(self, accuracy: float = None, top5_accuracy: float = None, loss: float = 0.0):
        """Track performance metrics - compatibility method."""
        pass  # Results tracked via track_evaluation_results instead
    
    def track_classification_report(self, y_true, y_pred, class_names=None):
        """Track classification report - compatibility method."""
        pass  # Not needed for JSON output
        
    def track_inference_latency(self, model, dataloader, num_samples: int = 100, 
                                 warmup_iters: int = 10):
        """Track detailed inference latency metrics with proper warmup and synchronization."""
        print(f"\n⏱️  Measuring Inference Latency ({num_samples} samples, {warmup_iters} warmup)...")
        
        model.eval()
        device = model.device if hasattr(model, 'device') else next(model.parameters()).device
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        
        latencies = []
        samples_processed = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # Extract images from batch
                if isinstance(batch, dict):
                    images = batch.get('images', batch.get('pixel_values'))
                elif isinstance(batch, (list, tuple)):
                    images = batch[0]
                else:
                    images = batch
                
                images = images.to(device)
                
                # Warmup phase
                if i < warmup_iters:
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    
                    try:
                        if hasattr(model, 'encode_image'):
                            _ = model.encode_image(images)
                        else:
                            _ = model(images)
                    except:
                        continue
                    
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    continue
                
                # Measurement phase
                if samples_processed >= num_samples:
                    break
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                start_time = time.perf_counter()
                
                try:
                    if hasattr(model, 'encode_image'):
                        _ = model.encode_image(images)
                    else:
                        _ = model(images)
                except:
                    continue
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                
                batch_size = images.shape[0]
                per_sample_latency_ms = latency_ms / batch_size
                
                latencies.append(per_sample_latency_ms)
                samples_processed += batch_size
        
        if len(latencies) == 0:
            print("❌ No latency measurements collected!")
            return
        
        latencies_array = np.array(latencies)
        
        self.metrics['latency'] = {
            'average_ms_per_sample': round(float(np.mean(latencies_array)), 4),
            'median_ms_per_sample': round(float(np.median(latencies_array)), 4),
            'std_ms_per_sample': round(float(np.std(latencies_array)), 4),
            'min_ms_per_sample': round(float(np.min(latencies_array)), 4),
            'max_ms_per_sample': round(float(np.max(latencies_array)), 4),
            'p95_ms_per_sample': round(float(np.percentile(latencies_array, 95)), 4),
            'p99_ms_per_sample': round(float(np.percentile(latencies_array, 99)), 4),
            'samples_measured': int(samples_processed),
            'warmup_iterations': int(warmup_iters),
            'throughput_samples_per_second': round(1000.0 / np.mean(latencies_array), 2)
        }
        
        print(f"   Average: {self.metrics['latency']['average_ms_per_sample']:.4f} ms/sample")
        print(f"   Throughput: {self.metrics['latency']['throughput_samples_per_second']:.2f} samples/sec")
        
        self.track_gpu_memory('inference')
        self.track_cpu_memory('inference')
        
    def track_evaluation_results(self, dataset_name: str, task: str, results: Dict[str, Any]):
        """Track evaluation results for a specific dataset and task."""
        if 'evaluation_results' not in self.metrics:
            self.metrics['evaluation_results'] = {}
        
        if dataset_name not in self.metrics['evaluation_results']:
            self.metrics['evaluation_results'][dataset_name] = {}
        
        self.metrics['evaluation_results'][dataset_name][task] = results
        
    def save_metrics(self, filename: str = "metrics.json"):
        """Save all metrics to a single JSON file."""
        if self.training_history['epochs']:
            self.metrics['training_history'] = self.training_history
        
        filepath = self.results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=4)
        
        print(f"\n💾 Saved all metrics to: {filepath}")
        
    def print_summary(self):
        """Print comprehensive metrics summary."""
        print(f"\n{'='*70}")
        print(f"METRICS SUMMARY: {self.model_name}")
        print(f"{'='*70}")
        
        if 'parameters' in self.metrics and self.metrics['parameters']:
            print(f"\n📊 PARAMETERS:")
            params = self.metrics['parameters']
            print(f"   Total: {params.get('total_parameters', 0):,}")
            print(f"   Trainable: {params.get('trainable_parameters', 0):,}")
        
        if 'memory' in self.metrics and self.metrics['memory']:
            print(f"\n💾 MEMORY:")
            for key, value in self.metrics['memory'].items():
                if value != 'N/A':
                    print(f"   {key}: {value:.2f} MB")
        
        if 'latency' in self.metrics and self.metrics['latency']:
            print(f"\n⏱️  LATENCY:")
            lat = self.metrics['latency']
            print(f"   Average: {lat.get('average_ms_per_sample', 0):.4f} ms/sample")
        
        print(f"\n{'='*70}\n")
