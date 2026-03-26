import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from models.frozen_clip import FrozenCLIP

class ModelEvaluator:
    def __init__(self, model, config, metrics_tracker, feature_cache):
        self.model = model
        self.config = config
        self.metrics = metrics_tracker
        self.cache = feature_cache
        self.device = config.device

    def evaluate_all(self, train_dataset, test_dataset, classnames, templates, dataset_name):
        # 1. Extract Features (Cached)
        train_features, train_labels = self.cache.get_features(train_dataset, f"{dataset_name}_train")
        test_features, test_labels = self.cache.get_features(test_dataset, f"{dataset_name}_test")

        # 2. Zero-Shot (Skip for Frozen)
        if self._supports_zero_shot():
            print(f"\n[1/3] Zero-Shot Evaluation")
            classifier = self._create_text_classifier(classnames, templates)
            zs_res = self._zero_shot_eval(test_features, test_labels, classifier)
            self.metrics.track_evaluation_results(dataset_name, 'zero_shot', zs_res)
            print(f"  Top-1: {zs_res['top1_accuracy']:.2f}%")
        else:
            print(f"\n[1/3] Zero-Shot Skipped (Generative Model)")

        # 3. Linear Probe
        print(f"\n[2/3] Linear Probe Evaluation")
        lp_res = self._linear_probe_eval(train_features, train_labels, test_features, test_labels)
        self.metrics.track_evaluation_results(dataset_name, 'linear_probe', lp_res)
        print(f"  Accuracy: {lp_res['accuracy']:.2f}%")

        # 4. Few-Shot (3 Trials)
        print(f"\n[3/3] Few-Shot Evaluation")
        fs_res = self._few_shot_eval(train_features, train_labels, test_features, test_labels)
        self.metrics.track_evaluation_results(dataset_name, 'few_shot', fs_res)

    def _supports_zero_shot(self):
        return not isinstance(self.model, FrozenCLIP) and hasattr(self.model, 'encode_text')

    def _create_text_classifier(self, classnames, templates):
        text_features = []
        with torch.no_grad():
            for classname in classnames:
                texts = [template.format(classname) for template in templates]
                tokens = self.model.tokenize(texts)
                embeddings = self.model.encode_text(tokens)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                text_features.append(embeddings.mean(dim=0))
        classifier = torch.stack(text_features).to(self.device)
        return classifier / classifier.norm(dim=-1, keepdim=True)

    def _zero_shot_eval(self, features, labels, classifier):
        features_t = torch.tensor(features).to(self.device)
        logits = features_t @ classifier.T
        preds_top1 = logits.argmax(dim=1).cpu().numpy()
        top1_acc = accuracy_score(labels, preds_top1) * 100
        preds_top5 = torch.topk(logits, k=5, dim=1)[1].cpu().numpy()
        top5_acc = np.mean([
                            label in preds_top5[i]
                            for i, label in enumerate(labels)
                            ]) * 100
        return {'top1_accuracy': float(top1_acc), 'top5_accuracy': float(top5_acc)}

    def _linear_probe_eval(self, X_train, y_train, X_test, y_test):
        import os
        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', n_jobs=min(8, os.cpu_count() or 1))
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        return {'accuracy': float(accuracy_score(y_test, preds) * 100), 'num_samples': len(y_test)}

    def _few_shot_eval(self, X_train, y_train, X_test, y_test):
        results = {}
        k_shots = getattr(self.config, 'k_shots', [1, 2, 4, 8, 16])
        num_trials = 3
        
        for k in k_shots:
            trial_accs = []
            for trial in range(num_trials):
                seed = 42 + trial
                np.random.seed(seed)
                indices = []
                for cls in np.unique(y_train):
                    cls_idx = np.where(y_train == cls)[0]
                    # Oversample if not enough samples
                    replace = len(cls_idx) < k
                    indices.extend(np.random.choice(cls_idx, k, replace=replace))
                
                if not indices: continue
                
                clf = LogisticRegression(max_iter=1000, solver='lbfgs', random_state=seed)
                clf.fit(X_train[indices], y_train[indices])
                acc = accuracy_score(y_test, clf.predict(X_test)) * 100
                trial_accs.append(acc)
            
            mean_acc = np.mean(trial_accs)
            results[f'{k}-shot'] = {
                'accuracy_mean': round(float(mean_acc), 2),
                'accuracy_std': round(float(np.std(trial_accs)), 2),
                'num_trials': num_trials
            }
            print(f"  {k}-shot: {mean_acc:.2f}%")
        return results