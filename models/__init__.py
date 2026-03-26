"""
Models module initialization.
"""

from models.clip_baseline import CLIPBaseline
from models.clip_lora import CLIPLoRA
from models.frozen_clip import FrozenCLIP

def get_model(model_name: str, config):
    """Factory function to get model by name."""
    if model_name == 'clip':
        return CLIPBaseline(config)
    elif model_name == 'clip_lora':
        return CLIPLoRA(config)
    elif model_name == 'frozen':
        return FrozenCLIP(config)
    else:
        raise ValueError(f"Unknown model: {model_name}")

__all__ = ['CLIPBaseline', 'CLIPLoRA', 'FrozenCLIP', 'get_model']
