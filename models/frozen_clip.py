"""
Frozen CLIP Model - Frozen language model with trainable vision encoder.
"""

import torch
import torch.nn as nn
import timm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from utils.helpers import pad_to_square

class VisionEncoder(nn.Module):
    """Vision encoder using ResNet-50 with projection."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pretrained ResNet50
        self.backbone = timm.create_model(
            config.vision_encoder_name,
            pretrained=True,
            num_classes=0,
            global_pool=''
        )
        
        # Global average pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Projection layer
        self.projection = nn.Linear(
            config.vision_hidden_dim,
            config.visual_prefix_length * config.lm_hidden_dim
        )
    
    def forward(self, images):
        features = self.backbone(images)  # [B, 2048, 7, 7]
        features = self.pool(features).flatten(1)  # [B, 2048]
        prefix = self.projection(features)  # [B, prefix_len * lm_dim]
        return prefix.view(-1, self.config.visual_prefix_length, self.config.lm_hidden_dim)


class FrozenCLIP(nn.Module):
    """Frozen architecture with trainable vision encoder and frozen LLM."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        # self.device = config.device
        
        # Vision encoder
        print(f"Loading vision encoder: {config.vision_encoder_name}...")
        self.vision_encoder = VisionEncoder(config).to(config.device)
        self.preprocess = transforms.Compose([
            transforms.Lambda(pad_to_square), 
            transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Language model
        print(f"Loading LLM: {config.language_model_name}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            config.language_model_name,
            cache_dir=config.cache_dir if hasattr(config, 'cache_dir') else None
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.language_model = GPT2LMHeadModel.from_pretrained(
            config.language_model_name,
            cache_dir=config.cache_dir if hasattr(config, 'cache_dir') else None
        )
        self.language_model.to(self.device)
        
        # Freeze language model
        for param in self.language_model.parameters():
            param.requires_grad = False
            
        # Print parameter counts
        frozen_params = sum(p.numel() for p in self.language_model.parameters())
        trainable_params = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)
        print(f"Frozen Params: {frozen_params:,} | Trainable Params: {trainable_params:,}")

    def tokenize(self, texts):
        """Tokenize texts for the frozen language model."""
        if isinstance(texts, str):
            texts = [texts]
            
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.visual_prefix_length if hasattr(self.config, 'visual_prefix_length') else 77,
            return_tensors='pt'
        )
        return {k: v.to(self.device) for k, v in encoded.items()}
    
    def forward(self, images, input_ids, attention_mask, labels=None):
        """Forward pass for training."""
        batch_size = images.size(0)
        
        # Encode images
        visual_prefix = self.vision_encoder(images)  # [B, prefix_len, hidden_dim]
        
        # Get text embeddings
        text_embeds = self.language_model.transformer.wte(input_ids)
        
        # Concatenate [visual_prefix, text]
        combined_embeds = torch.cat([visual_prefix, text_embeds], dim=1)
        
        # Adjust attention mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        visual_attention = torch.ones(
            batch_size, self.config.visual_prefix_length,
            dtype=torch.long,
            device=self.device
        )
        combined_mask = torch.cat([visual_attention, attention_mask], dim=1)
        
        # Forward through LM
        outputs = self.language_model(
            inputs_embeds=combined_embeds,
            attention_mask=combined_mask,
            return_dict=True
        )
        
        logits = outputs.logits
        
        # Compute loss manually if labels provided
        loss = None
        if labels is not None:
            # FIX: Start slicing one token earlier (prefix_length - 1)
            # This ensures the last Visual Token is used to predict the First Text Token.
            start_idx = self.config.visual_prefix_length - 1
            
            shift_logits = logits[:, start_idx:-1, :].contiguous()
            shift_labels = labels.contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return logits, loss
    
    def generate(self, images, tokenizer, max_length=50, temperature=1.0, top_k=50):
        """Generate captions for images using autoregressive sampling."""
        self.eval()
        batch_size = images.size(0)
        
        with torch.no_grad():
            # Get visual prefix
            visual_prefix = self.vision_encoder(images)
            
            # Start token
            start_token = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
            
            input_ids = torch.full(
                (batch_size, 1),
                start_token,
                dtype=torch.long,
                device=self.device
            )
            
            for _ in range(max_length):
                text_embeds = self.language_model.transformer.wte(input_ids)
                combined_embeds = torch.cat([visual_prefix, text_embeds], dim=1)
                
                outputs = self.language_model(
                    inputs_embeds=combined_embeds,
                    return_dict=True
                )
                
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('Inf')
                
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                if (next_token == tokenizer.eos_token_id).all():
                    break
            
            captions = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            
        return captions

    def encode_image(self, images):
        """Encode images to feature vectors for evaluation."""
        self.vision_encoder.eval()
        with torch.no_grad():
            features = self.vision_encoder(images) # [B, prefix_len, hidden_dim]
            
            # Mean pool across prefix tokens for consistent feature size [B, hidden_dim]
            if features.dim() == 3:
                features = features.mean(dim=1)
                
            features = features / features.norm(dim=-1, keepdim=True)
        return features
    
    def encode_text(self, text_tokens):
        """Encode text to feature vectors for evaluation."""
        self.language_model.eval()
        
        # --- FIX START: Handle Dictionary vs Tensor ---
        if isinstance(text_tokens, dict):
            # Extract inputs from Hugging Face dict
            input_ids = text_tokens['input_ids']
            # Use the mask provided by tokenizer, or generate if missing
            if 'attention_mask' in text_tokens:
                attention_mask = text_tokens['attention_mask']
            else:
                attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        else:
            # Fallback for raw tensors
            input_ids = text_tokens
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        # --- FIX END ---

        with torch.no_grad():
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            
            # Expand mask for broadcasting: [Batch, Seq_Len] -> [Batch, Seq_Len, 1]
            if attention_mask.dim() == 2:
                mask_expanded = attention_mask.unsqueeze(-1)
            else:
                mask_expanded = attention_mask

            # Weighted sum using the attention mask
            # Clamp sum to avoid division by zero
            sum_mask = mask_expanded.sum(dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)

            text_embeds = (hidden_states * mask_expanded).sum(dim=1) / sum_mask
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
            
        return text_embeds
    
    def eval(self):
        self.vision_encoder.eval()
        self.language_model.eval()
        return self
    
    def train(self, mode=True):
        self.vision_encoder.train(mode)
        self.language_model.eval()
        return self
    
    @property
    def device(self):
        return next(self.vision_encoder.parameters()).device

    def to(self, device):
        """Move model to device."""
        # Don't assign self.device - it's computed by @property
        self.vision_encoder = self.vision_encoder.to(device)
        self.language_model = self.language_model.to(device)
        return self

    def parameters(self):
        return self.vision_encoder.parameters()
    
    def count_parameters(self):
        """
        Count total and trainable parameters for Benchmark/Logging.
        Manually accesses submodules since parameters() is overridden.
        """
        # Vision Encoder (Trainable by default in this architecture)
        vision_total = sum(p.numel() for p in self.vision_encoder.parameters())
        vision_train = sum(p.numel() for p in self.vision_encoder.parameters() if p.requires_grad)

        # Language Model (Frozen by default in this architecture)
        lm_total = sum(p.numel() for p in self.language_model.parameters())
        lm_train = sum(p.numel() for p in self.language_model.parameters() if p.requires_grad)

        return {
            'total': vision_total + lm_total, 
            'trainable': vision_train + lm_train
        }

    def print_parameters(self):
        counts = self.count_parameters()
        print(f"FrozenCLIP Params -> Total: {counts['total']:,} | Trainable: {counts['trainable']:,}")