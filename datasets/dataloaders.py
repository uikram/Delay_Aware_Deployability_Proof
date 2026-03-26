"""
Unified dataset loading for all models.
"""

import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import CIFAR100, Food101, Flowers102, DTD, EuroSAT
from torchvision import transforms
from tqdm import tqdm
from utils.helpers import pad_to_square

Image.MAX_IMAGE_PIXELS = None

# ===== Conceptual Captions Dataset for CLIP =====

"""
Unified dataset loading for all models.
"""

import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import CIFAR100, Food101, Flowers102, DTD, EuroSAT
from torchvision import transforms
from tqdm import tqdm
from utils.helpers import pad_to_square

Image.MAX_IMAGE_PIXELS = None

# ===== Conceptual Captions Dataset for CLIP =====

class ConceptualCaptionsDataset(Dataset):
    """
    Conceptual Captions dataset for CLIP training.
    """
    
    def __init__(self, config, processor, image_dir=None, annotation_file=None, debug_mode=False, max_samples=None):
        # Allow overriding config paths for validation splits
        self.image_dir = Path(image_dir) if image_dir else Path(config.image_dir)
        annotation_path = Path(annotation_file) if annotation_file else Path(config.annotation_file)
        
        self.processor = processor
        self.max_length = config.max_length
        self.samples = []
        self.debug_mode = debug_mode
        
        print(f"Loading annotations from: {annotation_path}")
        print(f"Image directory: {self.image_dir}")
        
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Parse JSONL with robust error handling
        with open(annotation_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, raw_line in enumerate(tqdm(f, desc="Loading samples", disable=debug_mode), 1):
                if debug_mode and max_samples and len(self.samples) >= max_samples:
                    break
                
                line = raw_line.strip()
                if not line: continue
                
                try:
                    entry = json.loads(line)
                    # Handle multiple key variations
                    caption = entry.get("caption") or entry.get("text") or entry.get("description")
                    filepath = entry.get("filepath") or entry.get("image_path") or entry.get("file_name") or entry.get("image")
                    
                    if caption and filepath:
                        self.samples.append({"caption": caption, "image_path": filepath})
                except json.JSONDecodeError:
                    continue
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {annotation_path}. Check JSON format!")
        
        print(f"✓ Loaded {len(self.samples):,} samples")
        
        # --- Pre-flight Check ---
        if len(self.samples) > 0:
            # Check logic: Try full path first, then basename fallback
            first_item = self.samples[0]
            test_path = self.image_dir / first_item["image_path"]
            
            if not test_path.exists():
                # Try fallback
                fallback_path = self.image_dir / Path(first_item["image_path"]).name
                if fallback_path.exists():
                    print(f"ℹ Note: Using filename fallback logic. Found images at: {fallback_path}")
                else:
                    print(f"\n⚠️  WARNING: The first image was NOT found at:")
                    print(f"   {test_path}")
                    print(f"   Please check if 'validation_image_dir' in yaml matches your folder structure.")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample with INFINITE retry logic (until found)."""
        offset = 0
        
        # Infinite loop until a valid sample is found
        while True:
            current_idx = (idx + offset) % len(self.samples)
            item = self.samples[current_idx]
            
            # --- SMART PATH LOGIC ---
            # 1. Try appending the JSON path directly
            image_path = self.image_dir / item["image_path"]
            
            # 2. If not found, try using just the filename (fixes double-folder issues)
            if not image_path.exists():
                image_path = self.image_dir / Path(item["image_path"]).name
            
            try:
                if not image_path.exists():
                    offset += 1
                    continue
                
                image = Image.open(image_path).convert("RGB")
                inputs = self.processor(
                    text=[item["caption"]],
                    images=image,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length
                )
                
                return {
                    "pixel_values": inputs["pixel_values"].squeeze(0),
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0)
                }
            
            except Exception as e:
                # In debug mode, we might want to see why it failed
                if self.debug_mode and offset < 5:
                    print(f"Error loading {image_path}: {e}")
                
                # Move to next sample
                offset += 1
                continue

# ===== Conceptual Captions Dataset for Frozen Model =====

class FrozenConceptualCaptionsDataset(torch.utils.data.Dataset):
    """
    Dataset for Frozen model training (ResNet50 + GPT-2).
    """
    
    def __init__(self, image_dir, annotation_file, tokenizer, config, debug_mode=False, max_samples=None):
        from torchvision import transforms
        
        self.image_dir = Path(image_dir)
        self.tokenizer = tokenizer
        self.config = config
        self.debug_mode = debug_mode
        self.transform = transforms.Compose([
            transforms.Lambda(pad_to_square), 
            transforms.Resize((config.image_size, config.image_size), interpolation=transforms.InterpolationMode.BICUBIC), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Load annotations with robust error handling
        self.samples = []
        annotation_path = Path(annotation_file)
        
        print(f"[FrozenDataset] Loading annotations from: {annotation_path}")
        print(f"[FrozenDataset] Image directory: {self.image_dir}")
        
        if debug_mode:
            print("🔧 [FrozenDataset] DEBUG MODE: Loading limited samples for testing")
        
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
        
        if not self.image_dir.exists():
             raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        
        # Parse JSONL with on-the-fly corruption fixes
        error_count = 0
        
        with open(annotation_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, raw_line in enumerate(f, 1):
                # In debug mode, limit samples
                if debug_mode and max_samples and len(self.samples) >= max_samples:
                    print(f"[FrozenDataset] Reached max_samples limit: {max_samples}")
                    break
                
                # Clean the line
                line = raw_line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    
                    # Handle flexible key names
                    caption = (
                        entry.get("caption") or 
                        entry.get("text") or 
                        entry.get("description")
                    )
                    
                    rel_path = (
                        entry.get("filepath") or 
                        entry.get("image_path") or 
                        entry.get("file_name") or
                        entry.get("image")
                    )
                    
                    if caption and rel_path:
                        self.samples.append({
                            "caption": caption,
                            "image_path": rel_path
                        })
                    
                except json.JSONDecodeError as e:
                    error_count += 1
                    continue
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid samples found in {annotation_path}.")
        
        print(f"[FrozenDataset] ✓ Loaded {len(self.samples):,} samples ({error_count} JSON errors)")

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a single sample with retry logic."""
        attempts = 0
        # RESTORED: Original retry logic
        max_attempts = min(100, len(self.samples))
        
        while attempts < max_attempts:
            current_idx = (idx + attempts) % len(self.samples)
            item = self.samples[current_idx]
            image_path = self.image_dir / item["image_path"]
            
            try:
                if not image_path.exists():
                    attempts += 1
                    continue
                    
                # Load and transform image
                image = Image.open(image_path)
                if image.mode != 'RGB':
                    image = image.convert("RGB")
                    
                image = self.transform(image)
                
                # Tokenize caption with proper label masking
                caption_encoded = self.tokenizer(
                    item["caption"],
                    max_length=self.config.max_caption_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                input_ids = caption_encoded["input_ids"].squeeze(0)
                attention_mask = caption_encoded["attention_mask"].squeeze(0)
                
                # Mask padding tokens in labels
                labels = input_ids.clone()
                labels[attention_mask == 0] = -100  # Ignore padding in loss
                
                return {
                    "images": image,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels 
                }
                
            except Exception as e:
                attempts += 1
                continue
        
        raise RuntimeError(f"Failed to load sample after {max_attempts} attempts starting from index {idx}")


# ===== Dataset Factory for Benchmark Datasets =====

class DatasetFactory:
    """Factory for loading benchmark datasets."""
    
    @staticmethod
    def get_zeroshot_config(transform, data_root):
        """Get zero-shot evaluation datasets."""
        from utils.templates import (
            CIFAR100_CLASS_NAMES, FOOD101_CLASS_NAMES, FLOWERS102_CLASS_NAMES,
            DESCRIBEABLETEXTURES_CLASS_NAMES, EUROSAT_CLASS_NAMES,
            CIFAR100_TEMPLATES, FOOD101_TEMPLATES, FLOWERS102_TEMPLATES,
            DESCRIBEABLETEXTURES_TEMPLATES, EUROSAT_TEMPLATES
        )
        
        return {
            "CIFAR100": {
                "dataset": CIFAR100(root=data_root, download=True, train=False, transform=transform),
                "class_names": CIFAR100_CLASS_NAMES,
                "templates": CIFAR100_TEMPLATES
            },
            "Food101": {
                "dataset": Food101(root=data_root, download=True, split='test', transform=transform),
                "class_names": FOOD101_CLASS_NAMES,
                "templates": FOOD101_TEMPLATES
            },
            "Flowers102": {
                "dataset": Flowers102(root=data_root, download=True, split='test', transform=transform),
                "class_names": FLOWERS102_CLASS_NAMES,
                "templates": FLOWERS102_TEMPLATES
            },
            "DTD": {
                "dataset": DTD(root=data_root, download=True, split='test', transform=transform),
                "class_names": DESCRIBEABLETEXTURES_CLASS_NAMES,
                "templates": DESCRIBEABLETEXTURES_TEMPLATES
            },
            "EuroSAT": {
                "dataset": EuroSAT(root=data_root, download=True, transform=transform),
                "class_names": EUROSAT_CLASS_NAMES,
                "templates": EUROSAT_TEMPLATES
            }
        }
    
    @staticmethod
    def get_linear_probe_datasets(transform, data_root):
        """Get linear probe datasets."""
        return {
            "CIFAR100": {
                "train": CIFAR100(root=data_root, download=True, train=True, transform=transform),
                "test": CIFAR100(root=data_root, download=True, train=False, transform=transform)
            },
            "Food101": {
                "train": Food101(root=data_root, download=True, split='train', transform=transform),
                "test": Food101(root=data_root, download=True, split='test', transform=transform)
            },
            "Flowers102": {
                "train": Flowers102(root=data_root, download=True, split='train', transform=transform),
                "test": Flowers102(root=data_root, download=True, split='test', transform=transform)
            },
            "DTD": {
                "train": DTD(root=data_root, download=True, split='train', transform=transform),
                "test": DTD(root=data_root, download=True, split='test', transform=transform)
            },
            "EuroSAT": {
                "train": EuroSAT(root=data_root, download=True, transform=transform),
                "test": EuroSAT(root=data_root, download=True, transform=transform)
            }
        }


# ===== Dataloader Functions =====
# ===== Dataloader Function (Updated) =====

def get_conceptual_captions_loader(config, processor, split='train', debug_mode=False):
    max_samples = 500 if debug_mode else None
    
    if split == 'validation':
        if not hasattr(config, 'validation_file') or not config.validation_file:
            print("No validation file configured. Skipping validation loader.")
            return None
        image_dir = config.validation_image_dir
        annotation_file = config.validation_file
        shuffle = False
    else:
        image_dir = config.image_dir
        annotation_file = config.annotation_file
        shuffle = True

    dataset = ConceptualCaptionsDataset(
        config, 
        processor, 
        image_dir=image_dir,
        annotation_file=annotation_file,
        debug_mode=debug_mode, 
        max_samples=max_samples
    )
    
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers if not debug_mode else 0,
        pin_memory=True
    )
    
    return loader


def get_frozen_dataset_loader(config, debug_mode=False):
    """Get Conceptual Captions dataloaders for Frozen model."""
    from transformers import GPT2Tokenizer
    
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(config.language_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Limit samples in debug mode
    max_samples = 500 if debug_mode else None
    
    print("[FrozenDataLoader] Initializing training dataset...")
    # Create train dataset
    train_dataset = FrozenConceptualCaptionsDataset(
        config.train_image_dir,
        config.train_file,
        tokenizer,
        config,
        debug_mode=debug_mode,
        max_samples=max_samples
    )
    
    print("[FrozenDataLoader] Initializing validation dataset...")
    # Create val dataset (smaller in debug mode)
    val_max_samples = 100 if debug_mode else None
    val_dataset = FrozenConceptualCaptionsDataset(
        config.val_image_dir,
        config.val_file,
        tokenizer,
        config,
        debug_mode=debug_mode,
        max_samples=val_max_samples
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers if not debug_mode else 0,  # Single worker in debug
        pin_memory=True,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers if not debug_mode else 0,
        pin_memory=True,
        drop_last=False
    )
    
    print(f"[FrozenDataLoader] ✓ Train: {len(train_loader)} batches | Val: {len(val_loader)} batches")
    
    return train_loader, val_loader

def inspect_jsonl_file(filepath, max_lines=5):
    """Inspect JSONL file and print sample entries for debugging."""
    print(f"\n🔍 Inspecting: {filepath}")
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [line.strip() for line in f if line.strip()]
            print(f"  Total lines: {len(lines)}")
            
            if len(lines) == 0:
                print("File is empty!")
                return
            
            print(f"  First {min(max_lines, len(lines))} entries:")
            for i, line in enumerate(lines[:max_lines]):
                try:
                    entry = json.loads(line)
                    print(f"    [{i+1}] Keys: {list(entry.keys())}")
                    # Show first entry in detail
                    if i == 0:
                        for key, val in entry.items():
                            val_str = str(val)[:60] + "..." if len(str(val)) > 60 else str(val)
                            print(f"        {key}: {val_str}")
                except json.JSONDecodeError as e:
                    print(f"    [{i+1}] ✗ JSON Error: {str(e)[:50]}")
    except Exception as e:
        print(f"  ✗ Error reading file: {e}")