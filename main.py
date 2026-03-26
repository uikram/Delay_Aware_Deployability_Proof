"""
Main Entry Point.
Strictly separates Training, Evaluation, and Benchmarking modes.
"""
import argparse
import os
import torch
import warnings
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

from models import get_model
from utils.config import load_config_from_yaml
from utils.helpers import seed_everything
from evaluation.metrics import MetricsTracker
from training.train import get_trainer
from evaluation.evaluate import ModelEvaluator
from evaluation.feature_cache import FeatureCache
from utils.transforms import TransformFactory
from utils.templates import get_classnames, get_templates

BENCHMARK_DATASETS = ['cifar100', 'food101', 'flowers102', 'dtd', 'eurosat']

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Foundation Models")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'evaluate', 'benchmark'])
    parser.add_argument('--model', type=str, required=True, choices=['clip', 'clip_lora', 'frozen'])
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--datasets', nargs='+', default=BENCHMARK_DATASETS)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()

def init_environment(args):
    """Setup config, model, and metrics."""
    default_configs = {
        'clip': "configs/clip_baseline.yaml",
        'clip_lora': "configs/clip_lora.yaml",
        'frozen': "configs/frozen_clip.yaml"
    }
    config_path = args.config if args.config else default_configs[args.model]
    
    print(f"Loading config: {config_path}")
    config = load_config_from_yaml(config_path, args.model)
    config.device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    
    print(f"Initializing {args.model.upper()}")
    model = get_model(args.model, config)
    
    metrics = MetricsTracker(model_name=args.model.upper(), results_dir=Path(config.results_dir))
    return model, config, metrics

def load_checkpoint(model, model_name, config):
    """Restores specific logic for LoRA merging vs Adapters."""
    if model_name == 'clip_lora':
        from peft import PeftModel
        from models.clip_baseline import CLIPBaseline
        import copy
        
        checkpoint_dir = Path(config.output_dir)
        best_model_path = checkpoint_dir / "best_model"
        merged_path = checkpoint_dir / "merged_model"
        
        # # Priority 1: Best Model (Explicitly requested)
        # if best_model_path.exists():
        #     print(f"Loading LoRA adapter (Best Model): {best_model_path}")
        #     # FIX: Use load_adapter on the existing PeftModel instead of wrapping it again
        #     # This ensures we overwrite the random init weights with the trained ones
        #     try:
        #         model.model.load_adapter(best_model_path, adapter_name="best")
        #         model.model.set_adapter("best")
        #         return model
        #     except Exception as e:
        #         print(f"Error loading best model adapter: {e}")
        #         print("Falling back to other checkpoints...")

        # Priority 2: Merged Model
        if merged_path.exists() and (merged_path / "config.json").exists():
            print(f"Loading merged model: {merged_path}")
            merged_config = copy.copy(config)
            merged_config.model_id = str(merged_path)
            return CLIPBaseline(merged_config)
            
        # Priority 3: Latest Adapter
        epoch_dirs = sorted(checkpoint_dir.glob("epoch_*"), key=lambda p: int(p.name.split('_')[-1]))
        if epoch_dirs:
            latest = epoch_dirs[-1]
            print(f"Loading LoRA adapter: {latest}")
            # FIX: Use load_adapter here as well for consistency
            model.model.load_adapter(latest, adapter_name="latest")
            model.model.set_adapter("latest")
            
    elif model_name == 'frozen':
        ckpt = config.checkpoint_dir / "best_model.pt"
        if ckpt.exists():
            print(f"Loading checkpoint: {ckpt}")
            sd = torch.load(ckpt, map_location=config.device)
            state = sd['model_state'] if 'model_state' in sd else sd
            model.vision_encoder.load_state_dict(state, strict=False)
            
    return model

def load_dataset_helper(name, split, transform, config):
    """Handles EuroSAT missing split argument."""
    from datasets.benchmark_datasets import BenchmarkDatasets
    cache = config.cache_dir
    name = name.lower()
    
    if name == 'eurosat':
        # FIX: Pass the 'split' argument
        return BenchmarkDatasets.get_eurosat(cache, transform, split=split)
    
    loaders = {
        'cifar100': BenchmarkDatasets.get_cifar100,
        'food101': BenchmarkDatasets.get_food101,
        'flowers102': BenchmarkDatasets.get_flowers102,
        'dtd': BenchmarkDatasets.get_dtd,
    }
    return loaders[name](cache, transform, split)

# --- MODES ---

def run_train(args):
    model, config, metrics = init_environment(args)
    metrics.track_parameters(model)
    
    print(f"\n{'='*40}\nSTARTING TRAINING\n{'='*40}")
    trainer = get_trainer(model, config, metrics)
    trainer.train()
    metrics.save_metrics()
    print("Training Complete.")

def run_evaluate(args):
    model, config, metrics = init_environment(args)
    model = load_checkpoint(model, args.model, config)
    model.to(config.device)
    model.eval()
    
    transform = TransformFactory.get_transform(model)
    cache = FeatureCache(model, config)
    evaluator = ModelEvaluator(model, config, metrics, cache)
    
    print(f"\n{'='*40}\nSTARTING EVALUATION\n{'='*40}")
    for name in args.datasets:
        print(f"\nEvaluating on {name.upper()}")
        try:
            train_ds = load_dataset_helper(name, 'train', transform, config)
            test_ds = load_dataset_helper(name, 'test', transform, config)
            
            evaluator.evaluate_all(
                train_ds, test_ds, 
                get_classnames(name), 
                get_templates(name), 
                name
            )
        except Exception as e:
            print(f"Failed to evaluate {name}: {e}")
            import traceback; traceback.print_exc()

    metrics.save_metrics()
    print("Evaluation Complete.")

def run_benchmark(args):
    """
    Run benchmarking. 
    For CLIP LoRA, this runs TWO tests: Unmerged (Adapter) and Merged.
    For Frozen, it runs the standard best checkpoint.
    """
    # 1. SPECIAL HANDLING: CLIP LOORA (Run Both)
    if args.model == 'clip_lora':
        print(f"\n{'='*60}")
        print("BENCHMARK SUITE: CLIP LoRA (Adapter vs Merged)")
        print(f"{'='*60}")

        # --- A. Test Unmerged Adapter ---
        print("\n>>> TEST 1: Unmerged Adapter (PeftModel)")
        try:
            # Initialize fresh using the correct function name
            model, config, _ = init_environment(args)
            
            # Manual Adapter Load
            checkpoint_dir = Path(config.output_dir)
            epoch_dirs = sorted(checkpoint_dir.glob("epoch_*"), key=lambda p: int(p.name.split('_')[-1]))
            
            if epoch_dirs:
                latest = epoch_dirs[-1]
                print(f"Loading Adapter from: {latest}")
                from peft import PeftModel
                model.model = PeftModel.from_pretrained(model.model.base_model, latest)
                
                # Run Profile
                _execute_profile(model, "CLIP_LoRA_Adapter", config)
            else:
                print("No adapter checkpoints found (epoch_*). Skipping Adapter test.")
        except Exception as e:
            print(f"Adapter benchmark failed: {e}")
            import traceback; traceback.print_exc()

        # --- B. Test Merged Model ---
        print("\n>>> TEST 2: Merged Model (Standard Inference)")
        try:
            # Re-init just to get the clean config object
            _, config, _ = init_environment(args)
            checkpoint_dir = Path(config.output_dir)
            merged_path = checkpoint_dir / "merged_model"
            
            if merged_path.exists():
                print(f"Loading Merged Model from: {merged_path}")
                import copy
                from models.clip_baseline import CLIPBaseline
                
                # Create config pointing to merged path
                merged_config = copy.copy(config)
                merged_config.model_id = str(merged_path)
                
                # Load as standard CLIP
                merged_model = CLIPBaseline(merged_config)
                
                # Run Profile
                _execute_profile(merged_model, "CLIP_LoRA_Merged", merged_config)
            else:
                print("No 'merged_model' folder found. Skipping Merged test.")
        except Exception as e:
            print(f"Merged benchmark failed: {e}")
            import traceback; traceback.print_exc()

    # 2. STANDARD HANDLING: Frozen or CLIP Baseline
    else:
        model, config, _ = init_environment(args)
        
        # Load the specific checkpoint (best_model.pt for Frozen)
        model = load_checkpoint(model, args.model, config)
        
        _execute_profile(model, args.model.upper(), config)


def _execute_profile(model, run_name, config):
    """Helper to run the profiler on a model."""
    from evaluation.profiling import ModelProfiler
    from torch.utils.data import DataLoader
    from datasets.benchmark_datasets import BenchmarkDatasets
    from utils.transforms import TransformFactory
    
    model.to(config.device)
    model.eval()
    
    # 1. Get Transform
    if hasattr(model, 'preprocess'):
        transform = model.preprocess
    elif hasattr(model, 'processor'):
         def hf_transform(image):
            if image.mode != "RGB": image = image.convert("RGB")
            inputs = model.processor(images=image, return_tensors="pt")
            return inputs['pixel_values'].squeeze(0)
         transform = hf_transform
    else:
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(), 
            transforms.Normalize((0.481, 0.457, 0.408), (0.268, 0.261, 0.275))
        ])

    # 2. Load Data (CIFAR-100 Test)
    test_ds = BenchmarkDatasets.get_cifar100(config.cache_dir, transform, 'test')
    loader = DataLoader(test_ds, batch_size=config.batch_size, num_workers=config.num_workers)
    
    # 3. Profile
    profiler = ModelProfiler(model, run_name, config)
    results = profiler.profile(loader, num_samples=100) 
    profiler.save_results(results)
    profiler.print_summary(results)

def main():
    args = parse_args()
    if args.mode == 'train': run_train(args)
    elif args.mode == 'evaluate': run_evaluate(args)
    elif args.mode == 'benchmark': run_benchmark(args)

if __name__ == "__main__":
    main()