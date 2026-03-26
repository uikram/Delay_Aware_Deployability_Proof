"""
Trainer for CLIP + LoRA model.
"""

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os


class LoRATrainer:
    """Trainer for CLIP with LoRA adapters."""
    
    def __init__(self, model, config, metrics_tracker):
        self.model = model
        self.config = config
        self.metrics = metrics_tracker
        self.device = config.device
        
        # --- FIX: Use named_parameters() to properly filter weight decay ---
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight', 'ln.weight', 'ln.bias']
        
        # 1. Get all trainable parameters with their names
        trainable_params = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        
        # 2. Separate into decay (weights) and no_decay (biases/norms) groups
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in trainable_params if not any(nd in n for nd in no_decay)],
                'weight_decay': getattr(config, 'weight_decay', 0.01)
            },
            {
                'params': [p for n, p in trainable_params if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer, 
            T_max=config.num_epochs, 
            eta_min=config.learning_rate * 0.1
        )
        
        # Mixed Precision Scaler
        self.scaler = GradScaler()
        
        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
    
    def train(self):
        """Execute full training pipeline."""
        print("\n" + "="*60)
        print("CLIP + LoRA TRAINING")
        print("="*60)
        
        from datasets.dataloaders import get_conceptual_captions_loader
        
        # Load datasets
        print("Loading Conceptual Captions dataset (TRAIN)...")
        train_loader = get_conceptual_captions_loader(self.config, self.model.processor, split='train')
        
        print("Loading Conceptual Captions dataset (VALIDATION)...")
        val_loader = get_conceptual_captions_loader(self.config, self.model.processor, split='validation')
        
        if val_loader:
            print(f"✓ Loaded Val batches: {len(val_loader)}")
        else:
            print("⚠ No validation set found. Training will rely on training loss.")
        
        print(f"✓ Loaded {len(train_loader)} Train batches")
        print(f"  Batch size: {self.config.batch_size}")
        
        # Track memory before training
        self.metrics.track_gpu_memory('pre_training')
        
        # Start timer
        self.metrics.start_training_timer()
        
        # Training loop
        print(f"\nTraining for {self.config.num_epochs} epochs...")
        print("="*60)
        
        for epoch in range(self.config.num_epochs):
            # 1. Train
            train_loss = self.train_epoch(epoch, train_loader)
            
            # 2. Validate
            val_loss = self.validate(val_loader) if val_loader else train_loss
            
            # 3. Step Scheduler
            self.scheduler.step()
            
            # 4. Log
            self.metrics.track_epoch_metrics(epoch+1, train_loss=train_loss)
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs} Summary")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"{'='*60}\n")

            # 5. Save Checkpoint (using validation loss if available)
            save_metric = val_loss if val_loader else train_loss
            self.save_checkpoint(epoch, save_metric)
        
        # End timer
        self.metrics.end_training_timer()
        self.metrics.track_gpu_memory('post_training')
        self.metrics.track_performance(accuracy=0.0, loss=self.best_loss)
        
        print("\nTraining Complete!")
        print(f"Best Loss: {self.best_loss:.4f}")
    
    def train_epoch(self, epoch, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1} [Train]",
            dynamic_ncols=True
        )
        
        for batch in pbar:
            loss = self.train_step(batch)
            
            if not torch.isnan(torch.tensor(loss)):
                epoch_loss += loss
                batch_count += 1
            
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'avg': f'{epoch_loss/max(1, batch_count):.4f}'
            })
        
        return epoch_loss / batch_count if batch_count > 0 else 0.0
    
    def train_step(self, batch):
        """Single training step with AMP."""
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        self.optimizer.zero_grad()
        
        # Autocast for Mixed Precision
        with autocast(dtype=torch.float16):
            outputs = self.model.forward(**batch)
            loss = outputs.loss
        
        # Scale Loss & Clip Gradients
        self.scaler.scale(loss).backward()
        
        # Unscale before clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.global_step += 1
        return loss.item()

    def validate(self, val_loader):
        """Validation loop."""
        self.model.eval()
        val_loss = 0.0
        count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                with autocast(dtype=torch.float16):
                    outputs = self.model.forward(**batch)
                
                val_loss += outputs.loss.item()
                count += 1
                
        return val_loss / count if count > 0 else float('inf')
    
    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint safely."""
        try:
            checkpoint_path = self.config.output_dir / f"epoch_{epoch+1}"
            self.model.save_pretrained(checkpoint_path)
            
            # Save best model logic
            if loss < self.best_loss:
                self.best_loss = loss
                best_path = self.config.output_dir / "best_model"
                self.model.save_pretrained(best_path)
                print(f"★ New Best Model Saved (Loss: {loss:.4f})")
            
            print(f"Saved checkpoint to {checkpoint_path}")
            
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")