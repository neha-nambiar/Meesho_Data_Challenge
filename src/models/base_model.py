import gc
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from typing import Dict, Optional, Union, List
from sklearn.preprocessing import RobustScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from catboost import CatBoostClassifier, Pool
from sklearn.utils.class_weight import compute_class_weight



class BaseFashionModel:
    def __init__(self, num_attributes: int, attr_names: Optional[List[str]] = None):
        self.attribute_models = {}
        self.attributes = [f'attr_{i+1}' for i in range(num_attributes)]
        self.class_weights = {}
        self.n_attributes = num_attributes
        self.attr_names = attr_names if attr_names else self.attributes
        self.best_scores = {}
        self.attr_configs = {}  
        self.preprocessors = {}

    @staticmethod
    def calculate_attribute_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        return 2 * (macro_f1 * micro_f1) / (macro_f1 + micro_f1)

    def calculate_score(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[str, float]:
        scores = {}
        attribute_scores = []

        for i, attr_name in enumerate(self.attr_names):
            attr_col = f'attr_{i+1}'
            score = self.calculate_attribute_f1_score(
                y_true[attr_col].values,
                y_pred[attr_col].values
            )
            scores[attr_name] = score
            attribute_scores.append(score)

        scores['overall'] = np.mean(attribute_scores) if attribute_scores else 0.0
        return scores

    def preprocess_features(self, df: pd.DataFrame, is_training: bool = True) -> np.ndarray:
        """
        General preprocessing of image features. This implementation scales and applies PCA to all image feature columns.

        Parameters:
        - df: A pandas DataFrame containing the image feature columns.
        - is_training: If True, fit PCA and scalers; otherwise, transform using existing ones.

        Returns:
        - A numpy array containing stacked and processed feature vectors.
        """
        feature_columns = [col for col in df.columns if col.startswith("image_features_")]
        scaled_features = []
    
        for col in feature_columns:
            if is_training:
                scaler = RobustScaler()
                self.preprocessors[col] = scaler
                scaled_feature = scaler.fit_transform(np.vstack(df[col].values))
            else:
                if col not in self.preprocessors:
                    raise ValueError(f"Scaler for column '{col}' has not been fitted yet.")
                scaler = self.preprocessors[col]
                scaled_feature = scaler.transform(np.vstack(df[col].values))
    
            scaled_features.append(scaled_feature)
    
        return np.hstack(scaled_features)

    def _get_attribute_config(self, attr: str) -> dict:
        default_config = {
            'balance_strategy': 'class_weight',
            'model_params': {
                'depth': 6,
                'learning_rate': 0.1
            }
        }
        return self.attr_configs.get(attr, default_config)

    def _determine_sampling_strategy(self, y: np.ndarray) -> Dict:
        class_counts = np.bincount(y)
        max_count = np.max(class_counts)
        strategy = {i: max_count for i in range(len(class_counts))}
        return strategy

    def _apply_sampling_strategy(self, X: np.ndarray, y: np.ndarray, strategy: str, 
                               sampling_params: Optional[Dict] = None) -> tuple:
        if sampling_params is None:
            sampling_params = {}

        if strategy == 'smote':
            sampling_strategy = self._determine_sampling_strategy(y)
            sampler = SMOTE(sampling_strategy=sampling_strategy, random_state=42, **sampling_params)
            return sampler.fit_resample(X, y)
        elif strategy == 'smote_tomek':
            sampler = SMOTETomek(random_state=42)
            return sampler.fit_resample(X, y)
        elif strategy == 'undersample':
            sampling_strategy = 'auto'
            sampler = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42, **sampling_params)
            return sampler.fit_resample(X, y)
        else:  # No sampling
            return X, y

    def calculate_class_weights(self, y: np.ndarray, balance_strategy: str) -> Union[Dict, None]:
        if balance_strategy != 'class_weight':
            return None

        unique_classes = np.unique(y)
        if len(unique_classes) > 1:
            weights = compute_class_weight(
                class_weight='balanced',
                classes=unique_classes,
                y=y
            )
            return dict(zip(unique_classes, weights))
        return None

    def train(self, df: pd.DataFrame, validation_df: Optional[pd.DataFrame] = None, 
              epochs: int = 1000) -> 'BaseFashionModel':
        print("Preparing data for training...")
        X = self.preprocess_features(df)

        if validation_df is not None:
            X_val = self.preprocess_features(validation_df)

        print("Training models for each attribute...")
        for i, attr in enumerate(self.attributes, 1):
            print(f"\nTraining model for {attr}")

            config = self._get_attribute_config(attr)
            balance_strategy = config['balance_strategy']
            model_params = config['model_params']

            y = df[attr]
            mask = y.notna()
            X_attr = X[mask]
            y_attr = y[mask]

            X_balanced, y_balanced = self._apply_sampling_strategy(
                X_attr, y_attr, 
                balance_strategy,
                sampling_params={'k_neighbors': 5 if balance_strategy == 'smote' else None}
            )

            eval_dataset = None
            if validation_df is not None:
                y_val = validation_df[attr]
                val_mask = y_val.notna()
                X_val_attr = X_val[val_mask]
                y_val_attr = y_val[val_mask]
                eval_dataset = Pool(X_val_attr, y_val_attr)

            class_weights = self.calculate_class_weights(y_balanced, balance_strategy)

            base_params = {
                'iterations': epochs,
                'verbose': 200,
                'loss_function': 'MultiClass',
                'eval_metric': 'MultiClass',
                'custom_metric': ['F1'],
                'early_stopping_rounds': 100,
                'task_type': 'GPU',
                'nan_mode': 'Min'
            }

            model_params = {**base_params, **model_params}
            if class_weights is not None:
                model_params['class_weights'] = class_weights

            model = CatBoostClassifier(**model_params)

            if eval_dataset:
                model.fit(X_balanced, y_balanced, eval_set=eval_dataset)
            else:
                model.fit(X_balanced, y_balanced)

            self.attribute_models[attr] = model
            y_pred = model.predict(X_attr)
            score = self.calculate_attribute_f1_score(y_attr, y_pred)

            self.best_scores[attr] = score

            torch.cuda.empty_cache()
            gc.collect()
    
            print(f"Finished training {attr}. Best score: {score:.4f}")
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.preprocess_features(df)
        results = {attr: [] for attr in self.attributes}
        
        for attr in self.attributes:
            if attr in self.attribute_models:
                model = self.attribute_models[attr]
                predictions = model.predict(X).flatten()
                results[attr] = predictions.tolist()
        
        return pd.DataFrame(results)
