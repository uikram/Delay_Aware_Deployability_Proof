"""
Trainer for Frozen model (trainable vision encoder + frozen LLM).
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
import torch.nn as nn
from pathlib import Path

class FrozenTrainer:
    """Trainer for Frozen architecture."""
    
    def __init__(self, model, config, metrics_tracker):
        self.model = model
        self.config = config
        self.metrics = metrics_tracker
        self.device = config.device
        
        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(config.language_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Optimizer (Vision Encoder only)
        self.optimizer = torch.optim.Adam(
            self.model.vision_encoder.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay
        )
        
        # Mixed precision scaler
        self.scaler = torch.amp.GradScaler('cuda') if config.fp16 else None
        
        # Training state
        self.global_step = 0
        self.best_val_loss = float('inf')
    
    def train(self):
        """Execute full training pipeline."""
        print("\n" + "="*60)
        print("FROZEN MODEL TRAINING")
        print("="*60)
        
        # Load datasets
        train_loader, val_loader = self.get_dataloaders()
        
        print(f"✓ Loaded {len(train_loader)} training batches")
        print(f"✓ Loaded {len(val_loader)} validation batches")
        
        # Track memory
        self.metrics.track_gpu_memory('pre_training')
        self.metrics.start_training_timer()
        
        # Training loop
        print(f"\nTraining for {self.config.num_epochs} epochs...")
        print("="*60)
        
        for epoch in range(self.config.num_epochs):
            # Train for one epoch with mid-epoch validation
            train_loss = self.train_epoch(epoch, train_loader, val_loader)
            
            # End of epoch validation
            print(f"\nRunning End-of-Epoch Validation...")
            val_loss = self.validate(val_loader)
            self.metrics.track_epoch_metrics(epoch+1, train_loss=train_loss, val_loss=val_loss)
            
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{self.config.num_epochs} Complete")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Best Val Loss: {self.best_val_loss:.4f}")
            print(f"{'='*60}\n")
            
            # Save epoch checkpoint (Only once per epoch!)
            # Note: step_ckpt=False creates 'checkpoint_epoch_X.pt'
            self.save_checkpoint(epoch, val_loss, is_best=False, step_ckpt=False)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, is_best=True, step_ckpt=False)

        self.metrics.end_training_timer()
        self.metrics.track_gpu_memory('post_training')
        print("\nTraining Complete!")
        
        return self.model
        
    def get_dataloaders(self):
        from datasets.dataloaders import FrozenConceptualCaptionsDataset
        
        debug_mode = getattr(self.config, 'debug_mode', False)
        max_samples = 500 if debug_mode else None
        
        train_ds = FrozenConceptualCaptionsDataset(
            Path(self.config.train_image_dir),
            Path(self.config.train_file),
            self.tokenizer,
            self.config,
            debug_mode=debug_mode,
            max_samples=max_samples
        )
        
        val_ds = FrozenConceptualCaptionsDataset(
            Path(self.config.val_image_dir),
            Path(self.config.val_file),
            self.tokenizer,
            self.config,
            debug_mode=debug_mode,
            max_samples=100 if debug_mode else None
        )
        
        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0 if debug_mode else self.config.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0 if debug_mode else self.config.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader

    def train_epoch(self, epoch, train_loader, val_loader):
        """Train for one epoch with mid-epoch validation."""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            dynamic_ncols=True
        )
        
        for step, batch in enumerate(pbar):
           
            initial_global_step = self.global_step
            
            loss = self.train_step(batch, step)
            epoch_loss += loss
            pbar.set_postfix({'loss': f'{loss:.4f}', 'step': self.global_step})
            
            if self.global_step > initial_global_step and self.global_step % 1000 == 0:
                print(f"\n\n[Step {self.global_step}] Running Mid-Epoch Validation...")
                val_loss = self.validate(val_loader)
                print(f"[Step {self.global_step}] Val Loss: {val_loss:.4f}\n")
                
                # Save step checkpoint (step_ckpt=True creates 'checkpoint_step_X.pt')
                self.save_checkpoint(epoch, val_loss, is_best=False, step_ckpt=True)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, val_loss, is_best=True, step_ckpt=True)
                
                # Ensure model is back in train mode
                self.model.train()
        
        return epoch_loss / len(train_loader)
    
    def train_step(self, batch, step):
        """Single training step."""
        images = batch['images'].to(self.device)
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        
        # Forward pass
        if self.config.fp16:
            with torch.amp.autocast('cuda'):
                logits, loss = self.model.forward(
                    images, input_ids, attention_mask, labels
                )
                loss = loss / self.config.gradient_accumulation_steps
        else:
            logits, loss = self.model.forward(
                images, input_ids, attention_mask, labels
            )
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Optimization step - Triggered by Loop Index 'step'
        if (step + 1) % self.config.gradient_accumulation_steps == 0:
            if self.config.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            self.optimizer.zero_grad()
            self.global_step += 1 # Increment global step
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    def validate(self, val_loader):
        """Validation loop."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                images = batch['images'].to(self.device)
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                if self.config.fp16:
                    with torch.amp.autocast('cuda'):
                        _, loss = self.model.forward(
                            images, input_ids, attention_mask, labels
                        )
                else:
                    _, loss = self.model.forward(
                        images, input_ids, attention_mask, labels
                    )
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def save_checkpoint(self, epoch, val_loss, is_best=False, step_ckpt=False):
        """Save checkpoint with clear distinction between step and epoch checkpoints."""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state': self.model.vision_encoder.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'config': vars(self.config)
        }
        
        # 1. Step Checkpoint (e.g., checkpoint_step_1000.pt)
        if step_ckpt:
            step_path = self.config.checkpoint_dir / f"checkpoint_step_{self.global_step}.pt"
            torch.save(checkpoint, step_path)
            print(f"Saved STEP checkpoint: {step_path.name}")
        
        # 2. Epoch Checkpoint (e.g., checkpoint_epoch_1.pt)
        else:
            epoch_path = self.config.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint, epoch_path)
            print(f"Saved EPOCH checkpoint: {epoch_path.name}")
        
        # 3. Best Model (Always save if best)
        if is_best:
            best_path = self.config.checkpoint_dir / "best_model_frozen.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved BEST model (val_loss: {val_loss:.4f})")