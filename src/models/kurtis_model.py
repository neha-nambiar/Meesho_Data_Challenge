from src.models.base_model import BaseFashionModel


class Kurtis_Model(BaseFashionModel):
    def __init__(self):
        """
        Initializes the Kurtis model with specific attribute configurations.
        """
        super().__init__(num_attributes=9)
        self.attr_configs = {
            'attr_1': {
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 7,
                    'learning_rate': 0.08,
                    'l2_leaf_reg': 3,
                    'random_strength': 1
                }
            },
            'attr_2': {  # Binary classification with imbalance
                'balance_strategy': 'smote_tomek',
                'model_params': {
                    'depth': 6,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 5,
                    'random_strength': 0.8
                }
            },
            'attr_3': {  # Binary classification with imbalance
                'balance_strategy': 'smote_tomek',
                'model_params': {
                    'depth': 6,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 5,
                    'random_strength': 0.8
                }
            },
            'attr_4': {  # More balanced binary classification
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 6,
                    'learning_rate': 0.1,
                    'l2_leaf_reg': 3,
                    'random_strength': 1
                }
            },
            'attr_5': {  # Binary classification with imbalance
                'balance_strategy': 'smote_tomek',
                'model_params': {
                    'depth': 6,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 5,
                    'random_strength': 0.8
                }
            },
            'attr_6': {
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 7,
                    'learning_rate': 0.08,
                    'l2_leaf_reg': 3,
                    'random_strength': 1
                }
            },
            'attr_7': {
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 7,
                    'learning_rate': 0.08,
                    'l2_leaf_reg': 3,
                    'random_strength': 1
                }
            },
            'attr_8': {  # Sparse multi-class problem
                'balance_strategy': 'smote',
                'model_params': {
                    'depth': 8,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 7,
                    'random_strength': 1.2,
                    'min_data_in_leaf': 5
                }
            },
            'attr_9': {  # Binary classification
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 6,
                    'learning_rate': 0.1,
                    'l2_leaf_reg': 3,
                    'random_strength': 1
                }
            }
        }
