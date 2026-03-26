"""
Trainer Factory: Dispatches to specialized trainers.
"""
def get_trainer(model, config, metrics_tracker):
    # 1. Frozen Model
    if hasattr(config, 'vision_encoder_name') or getattr(config, 'model_name', '') == 'frozen':
        from training.trainer_frozen import FrozenTrainer
        return FrozenTrainer(model, config, metrics_tracker)
    
    # 2. CLIP LoRA
    elif hasattr(config, 'lora_r') or getattr(config, 'model_name', '') == 'clip_lora':
        from training.trainer_lora import LoRATrainer
        return LoRATrainer(model, config, metrics_tracker)
    
    # 3. CLIP Baseline
    elif getattr(config, 'model_name', '') == 'clip':
        from training.trainer_clip import CLIPTrainer
        return CLIPTrainer(model, config, metrics_tracker)
        
    else:
        raise ValueError(f"No suitable trainer found for config: {config}")