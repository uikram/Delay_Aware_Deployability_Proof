import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

class FeatureCache:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.cache = {}
        self.device = config.device

    def get_features(self, dataset, cache_key):
        if cache_key in self.cache:
            print(f"Using cached features for {cache_key}")
            return self.cache[cache_key]
        
        print(f"Extracting features for {cache_key}...")
        features, labels = self._extract(dataset)
        self.cache[cache_key] = (features, labels)
        return features, labels
    
    def _extract(self, dataset):
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, 
                            num_workers=self.config.num_workers, pin_memory=True)
        all_features, all_labels = [], []
        
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc="Extracting", leave=False):
                if isinstance(batch, (list, tuple)):
                    imgs, lbls = batch
                else:
                    imgs, lbls = batch['images'], batch['labels']
                
                imgs = imgs.to(self.device)
                if hasattr(self.model, 'encode_image'):
                    feats = self.model.encode_image(imgs)
                else:
                    feats = self.model(imgs)
                
                all_features.append(feats.cpu().numpy())
                if lbls is not None: all_labels.append(lbls.numpy())
                
        return np.vstack(all_features), np.concatenate(all_labels) if all_labels else None