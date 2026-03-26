"""
Trainer for CLIP Baseline model.

CLIP baseline is pretrained, so this module handles evaluation workflows only.
"""

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LogisticRegression


class CLIPTrainer:
    """
    Trainer for CLIP Baseline (pretrained model).
    
    Since CLIP is pretrained, this trainer focuses on:
    - Feature extraction
    - Linear probe training
    - Zero-shot evaluation
    - Few-shot evaluation
    """
    
    def __init__(self, model, config, metrics_tracker):
        self.model = model
        self.config = config
        self.metrics = metrics_tracker
        self.device = config.device
    
    def train(self):
        """
        CLIP baseline is pretrained - no training needed.
        This method performs evaluation instead.
        """
        print("\n" + "="*60)
        print("CLIP BASELINE - PRETRAINED MODEL")
        print("="*60)
        print("ℹ️  CLIP baseline uses pretrained weights.")
        print("   Skipping training phase.")
        print("   Use evaluation mode for benchmarking.")
        print("="*60 + "\n")
        
        # Track that training was skipped
        self.metrics.metrics['training_time'] = {
            'total_seconds': 0,
            'total_hours': 0,
            'note': 'Pretrained model - no training required'
        }
    
    def extract_features(self, dataset):
        """Extract CLIP features from a dataset."""
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        all_features = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Extracting features"):
                images = images.to(self.device)
                features = self.model.encode_image(images)
                features /= features.norm(dim=-1, keepdim=True)
                
                all_features.append(features.cpu())
                all_labels.append(labels.cpu())
        
        return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()
    
    def linear_probe(self, train_dataset, test_dataset):
        """
        Perform linear probe evaluation.
        
        Extracts features from frozen CLIP and trains a linear classifier.
        """
        print("\n" + "-"*60)
        print("Linear Probe Evaluation")
        print("-"*60)
        
        # Extract features
        print("Extracting training features...")
        train_features, train_labels = self.extract_features(train_dataset)
        
        print("Extracting test features...")
        test_features, test_labels = self.extract_features(test_dataset)
        
        # Train logistic regression
        print("Training logistic regression classifier...")
        classifier = LogisticRegression(
            random_state=self.config.seed,
            C=self.config.logistic_regression_c,
            max_iter=1000,
            verbose=1,
            n_jobs=-1,
            solver='lbfgs'
        )
        classifier.fit(train_features, train_labels)
        
        # Evaluate
        print("Evaluating...")
        predictions = classifier.predict(test_features)
        accuracy = np.mean((test_labels == predictions).astype(float)) * 100.0
        
        print(f"✓ Linear Probe Accuracy: {accuracy:.2f}%")
        
        return accuracy, predictions, test_labels
    
    def zero_shot(self, dataset, text_classifier):
        """
        Perform zero-shot classification.
        
        Args:
            dataset: Image dataset
            text_classifier: Pre-computed text embeddings (class prototypes)
        """
        print("\n" + "-"*60)
        print("Zero-Shot Evaluation")
        print("-"*60)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        all_predictions = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Zero-shot evaluation"):
                images = images.to(self.device)
                
                # Encode images
                image_features = self.model.encode_image(images)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Compute similarities with text prototypes
                logits = (100.0 * image_features @ text_classifier.T)
                _, predictions = logits.max(1)
                
                all_predictions.append(predictions.cpu())
                all_labels.append(labels.cpu())
        
        all_predictions = torch.cat(all_predictions).numpy()
        all_labels = torch.cat(all_labels).numpy()
        
        accuracy = np.mean((all_labels == all_predictions).astype(float)) * 100.0
        print(f"✓ Zero-Shot Accuracy: {accuracy:.2f}%")
        
        return accuracy, all_predictions, all_labels
    
    def few_shot(self, train_dataset, test_dataset, k_shots=None):
        """
        Perform few-shot evaluation.
        
        Args:
            train_dataset: Training dataset for sampling few-shot examples
            test_dataset: Test dataset
            k_shots: List of k values (e.g., [1, 2, 4, 8, 16])
        """
        if k_shots is None:
            k_shots = self.config.k_shots
        
        print("\n" + "-"*60)
        print("Few-Shot Evaluation")
        print("-"*60)
        
        # Extract features once
        print("Caching features...")
        train_features, train_labels = self.extract_features(train_dataset)
        test_features, test_labels = self.extract_features(test_dataset)
        
        results = {}
        unique_classes = np.unique(train_labels)
        
        for k in k_shots:
            print(f"\nEvaluating {k}-shot...")
            
            # Sample k examples per class
            indices = []
            for c in unique_classes:
                c_indices = np.where(train_labels == c)[0]
                n_samples = min(len(c_indices), k)
                
                if n_samples > 0:
                    np.random.seed(self.config.seed)  # Reproducibility
                    chosen = np.random.choice(c_indices, n_samples, replace=False)
                    indices.extend(chosen)
            
            if len(indices) == 0:
                print(f"  Skipping {k}-shot: No samples available")
                continue
            
            # Train classifier on k-shot samples
            k_features = train_features[indices]
            k_labels = train_labels[indices]
            
            try:
                clf = LogisticRegression(
                    random_state=self.config.seed,
                    C=self.config.logistic_regression_c,
                    max_iter=1000,
                    verbose=0,
                    n_jobs=-1,
                    solver='lbfgs'
                )
                clf.fit(k_features, k_labels)
                
                # Evaluate
                predictions = clf.predict(test_features)
                accuracy = np.mean((test_labels == predictions).astype(float)) * 100.0
                
                results[f"{k}-shot"] = {
                    'accuracy': accuracy,
                    'predictions': predictions
                }
                
                print(f"{k}-shot Accuracy: {accuracy:.2f}%")
                
            except Exception as e:
                print(f"{k}-shot Failed: {e}")
        
        return results, test_labels
    
    def create_text_classifier(self, class_names, templates):
        """
        Create text classifier from class names and prompt templates.
        
        Returns normalized text embeddings for zero-shot classification.
        """
        all_text_features = []
        
        self.model.eval()
        with torch.no_grad():
            for class_name in tqdm(class_names, desc="Encoding text prompts"):
                # Generate prompts from templates
                texts = [template.format(class_name) for template in templates]
                
                # Tokenize and encode
                text_tokens = self.model.tokenizer(texts).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                
                # Ensemble over templates (average)
                text_features = text_features.mean(dim=0)
                text_features /= text_features.norm()
                
                all_text_features.append(text_features)
        
        return torch.stack(all_text_features, dim=0)
