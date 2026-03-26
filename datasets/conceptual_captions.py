"""
Conceptual Captions dataset implementation.
"""

import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import warnings

# Suppress all warnings from this module
warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None


class ConceptualCaptionsDataset(Dataset):
    """
    Conceptual Captions dataset for CLIP LoRA training.
    
    Expects JSONL format with entries like:
    {"caption": "A cat sitting on a mat", "filepath": "images/001.jpg"}
    or
    {"text": "A cat sitting on a mat", "file_name": "train/001.jpg"}
    """
    
    def __init__(self, config, processor):
        self.image_dir = Path(config.image_dir)  # Ensure it's a Path object
        self.processor = processor
        self.max_length = config.max_length
        self.samples = []
        
        print(f"Loading annotations from: {config.annotation_file}")
        print(f"Image directory: {self.image_dir}")
        
        annotation_file = Path(config.annotation_file)
        
        if not annotation_file.exists():
            raise FileNotFoundError(
                f"Annotation file not found: {annotation_file}"
            )
        
        # Parse JSONL file
        with open(annotation_file, 'r') as f:
            for line_num, line in enumerate(tqdm(f, desc="Parsing JSONL"), 1):
                line = line.strip()
                if not line:  # Skip empty lines
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
                            "image_path": rel_path  # Keep as string, will convert later
                        })
                    else:
                        if line_num <= 10:  # Only warn for first 10
                            print(f"Warning: Skipping line {line_num}, missing caption or image path")
                
                except json.JSONDecodeError as e:
                    if line_num <= 10:
                        print(f"Warning: Failed to parse line {line_num}: {e}")
                    continue
        
        if len(self.samples) == 0:
            raise ValueError("No valid samples found in dataset!")
        
        print(f"✓ Loaded {len(self.samples):,} valid samples")
        
        # Verify first image exists
        first_sample_path = self.image_dir / self.samples[0]["image_path"]
        if first_sample_path.exists():
            print(f"✓ First image verified: {first_sample_path}")
        else:
            print(f"⚠️  Warning: First image not found: {first_sample_path}")
            print(f"   Check if image_dir ({self.image_dir}) is correct")
        self.skip_count = 0
        self.skip_warn_threshold = 1000  # Print every 1000 skips (less spam)
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Get item with INFINITE retry - never crashes on bad images.
        
        Keeps trying different samples until it finds a valid one.
        This ensures training NEVER stops due to corrupt/missing images.
        """
        attempts = 0
        max_attempts = len(self.samples)  # Try entire dataset if needed
        
        while attempts < max_attempts:
            current_idx = (idx + attempts) % len(self.samples)
            item = self.samples[current_idx]
            
            # Proper path joining
            image_path = self.image_dir / item["image_path"]
            
            try:
                # Check file exists
                if not image_path.exists():
                    attempts += 1
                    self.skip_count += 1
                    if self.skip_count % self.skip_warn_threshold == 0:
                        print(f"[Dataset] Skipped {self.skip_count} missing/corrupt images so far...")
                    continue
                
                # Load image
                image = Image.open(image_path)
                
                # Force RGB conversion
                if image.mode != 'RGB':
                    image = image.convert("RGB")
                
                # Double-check it's RGB
                if image.mode != 'RGB':
                    attempts += 1
                    self.skip_count += 1
                    continue
                
                # Process with CLIP processor
                inputs = self.processor(
                    text=[item["caption"]],
                    images=image,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_length
                )
                
                # Return successfully
                return {
                    "pixel_values": inputs["pixel_values"].squeeze(0),
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0)
                }
            
            except Exception as e:
                # Catch ALL errors and just skip to next sample
                # SILENT - no printing individual errors
                attempts += 1
                self.skip_count += 1
                
                # Only print summary every 100 skips
                if self.skip_count % self.skip_warn_threshold == 0:
                    print(f"[Dataset] Skipped {self.skip_count} images total")
                
                continue
        
        # Fallback dummy sample (very unlikely to reach here)
        dummy_image = torch.zeros(3, 224, 224)
        dummy_text = "empty"
        
        inputs = self.processor(
            text=[dummy_text],
            images=Image.new('RGB', (224, 224)),
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