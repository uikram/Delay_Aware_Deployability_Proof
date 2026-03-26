"""
Utility for merging LoRA adapters into base CLIP model.
Required for fair evaluation latency and memory measurements.
"""

import torch
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor
from peft import PeftModel


class LoRAModelMerger:
    """
    Handles merging of LoRA adapters into base CLIP model.
    Ensures evaluation measurements are not affected by adapter hook overhead.
    """
    
    @staticmethod
    def merge_and_save_lora(
        base_model_id: str,
        lora_checkpoint_dir: Path,
        output_dir: Path,
        cache_dir: Path = None
    ):
        """
        Merge LoRA adapters into base CLIP model and save.
        
        Args:
            base_model_id: Hugging Face model ID for base CLIP
            lora_checkpoint_dir: Directory containing LoRA adapters
            output_dir: Directory to save merged model
            cache_dir: Cache directory for downloading models
            
        Returns:
            Path to merged model directory
        """
        print(f"\n🔧 Merging LoRA adapters into base CLIP model...")
        print(f"   Base model: {base_model_id}")
        print(f"   LoRA checkpoint: {lora_checkpoint_dir}")
        
        # Load base CLIP model
        base_model = CLIPModel.from_pretrained(
            base_model_id,
            cache_dir=cache_dir,
            torch_dtype=torch.float32  # Use FP32 for merging
        )
        
        # Load PEFT model with adapters
        peft_model = PeftModel.from_pretrained(
            base_model,
            str(lora_checkpoint_dir),
            is_trainable=False
        )
        
        print("   Loaded PEFT model with adapters")
        
        # Merge adapters into base weights
        merged_model = peft_model.merge_and_unload()
        print("   ✅ Merged adapters into base model")
        
        # Save merged model
        output_dir.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(output_dir))
        
        # Also save processor
        processor = CLIPProcessor.from_pretrained(base_model_id, cache_dir=cache_dir)
        processor.save_pretrained(str(output_dir))
        
        print(f"   💾 Saved merged model to: {output_dir}")
        
        return output_dir
    
    @staticmethod
    def load_merged_lora_model(merged_model_dir: Path, device: str = 'cuda'):
        """
        Load a previously merged LoRA+CLIP model.
        
        Args:
            merged_model_dir: Directory containing merged model
            device: Device to load model on
            
        Returns:
            Tuple of (model, processor)
        """
        print(f"\n📂 Loading merged CLIP+LoRA model from: {merged_model_dir}")
        
        model = CLIPModel.from_pretrained(
            str(merged_model_dir),
            torch_dtype=torch.float16
        ).to(device)
        
        processor = CLIPProcessor.from_pretrained(str(merged_model_dir))
        
        print("   ✅ Loaded merged model (no adapter hooks)")
        
        return model, processor
    
    @staticmethod
    def find_latest_lora_checkpoint(lora_output_dir: Path):
        """
        Find the latest LoRA checkpoint in output directory.
        
        Args:
            lora_output_dir: Base output directory for LoRA training
            
        Returns:
            Path to latest checkpoint directory
        """
        # Look for epoch_X directories
        epoch_dirs = sorted(lora_output_dir.glob("epoch_*"))
        
        if not epoch_dirs:
            raise FileNotFoundError(f"No LoRA checkpoints found in {lora_output_dir}")
        
        latest_checkpoint = epoch_dirs[-1]
        print(f"   Found latest checkpoint: {latest_checkpoint}")
        
        return latest_checkpoint
