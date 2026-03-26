"""
CLIP Baseline Model - Standard CLIP without modifications.
"""

import torch
from transformers import CLIPModel, CLIPProcessor

class CLIPBaseline:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # FIX: Use model_id (not model_name)
        model_id = getattr(config, 'model_id', getattr(config, 'model_name', 'openai/clip-vit-base-patch32'))
        
        print(f"Loading CLIP Baseline (Hugging Face): {model_id}")
        
        # Load processor
        self.processor = CLIPProcessor.from_pretrained(
            model_id,
            cache_dir=config.cache_dir
        )
        
        # Load model
        self.model = CLIPModel.from_pretrained(
            model_id,
            cache_dir=config.cache_dir,
            torch_dtype=torch.float16
        )
        
        self.model.to(self.device)
        self.model.eval()

    def tokenize(self, texts):
        """Tokenize texts using the model's processor."""
        inputs = self.processor(
            text=texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
        
    def forward(self, **kwargs):
        """Forward pass through the model."""
        return self.model(**kwargs)
    
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train()
        
    def parameters(self):
        return self.model.parameters()
    
    def count_parameters(self):
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

    def to(self, device):
        self.device = device
        self.model = self.model.to(device)
        return self
    
    def encode_image(self, images):
        """Encode images."""
        vision_outputs = self.model.get_image_features(pixel_values=images)
        return vision_outputs
    
    def encode_text(self, tokenized_inputs):
        """Encode text using input_ids and attention_mask."""
        if isinstance(tokenized_inputs, dict):
            return self.model.get_text_features(**tokenized_inputs)
        else:
            return self.model.get_text_features(input_ids=tokenized_inputs)
