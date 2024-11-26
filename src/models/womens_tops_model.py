from src.models.base_model import BaseFashionModel

class Women_Tops_Model(BaseFashionModel):
    def __init__(self):
        """
        Initializes the Women Tops & Tunics model with specific attribute configurations.
        """
        super().__init__(num_attributes=10)
        self.attr_configs = {
            'attr_1': {
                'balance_strategy': 'smote',
                'model_params': {
                    'depth': 8,
                    'learning_rate': 0.08,
                    'l2_leaf_reg': 5
                }
            },
            'attr_2': {
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 7,
                    'learning_rate': 0.09,
                    'l2_leaf_reg': 3
                }
            },
            'attr_3': {
                'balance_strategy': 'smote',
                'model_params': {
                    'depth': 6,
                    'learning_rate': 0.1,
                    'l2_leaf_reg': 4  
                }
            },
            'attr_4': {
                'balance_strategy': 'hybrid',
                'model_params': {
                    'depth': 9,
                    'learning_rate': 0.07,
                    'random_strength': 1
                }
            },
            'attr_5': {
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 8,
                    'learning_rate': 0.085,
                    'random_strength': 0.8
                }
            },
            'attr_6': {
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 7,
                    'learning_rate': 0.09,
                    'l2_leaf_reg': 4
                }
            },
            'attr_7': {
                'balance_strategy': 'hybrid',
                'model_params': {
                    'depth': 8,
                    'learning_rate': 0.075,
                    'random_strength': 1.2
                }
            },
            'attr_8': {
                'balance_strategy': 'smote',
                'model_params': {
                    'depth': 7,
                    'learning_rate': 0.085,
                    'l2_leaf_reg': 3
                }
            },
            'attr_9': {
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 8,
                    'learning_rate': 0.08,
                    'random_strength': 0.9
                }
            },
            'attr_10': {
                'balance_strategy': 'hybrid',
                'model_params': {
                    'depth': 7,
                    'learning_rate': 0.09,
                    'l2_leaf_reg': 4
                }
            }
        }