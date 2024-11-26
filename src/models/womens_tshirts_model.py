from src.models.base_model import BaseFashionModel

class Women_Tshirts_Model(BaseFashionModel):
    def __init__(self):
        """
        Initializes the Women T-shirts model with specific attribute configurations.
        """
        super().__init__(num_attributes=8)
        self.attr_configs = {
            'attr_1': {  
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 8,
                    'learning_rate': 0.07,
                    'l2_leaf_reg': 4,
                    'random_strength': 0.8,
                    'bootstrap_type': 'Bernoulli',
                    'subsample': 0.8
                }
            },
            'attr_2': {  
                'balance_strategy': 'smote',
                'model_params': {
                    'depth': 7,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 5,
                    'random_strength': 1.0,
                    'bootstrap_type': 'Bernoulli',
                    'subsample': 0.85
                }
            },
            'attr_3': {  
                'balance_strategy': 'smote',
                'model_params': {
                    'depth': 6,
                    'learning_rate': 0.08,
                    'l2_leaf_reg': 3,
                    'random_strength': 0.5
                }
            },
            'attr_4': { 
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 6,
                    'learning_rate': 0.1,
                    'l2_leaf_reg': 2,
                    'random_strength': 0.3
                }
            },
            'attr_5': { 
                'balance_strategy': 'smote',
                'model_params': {
                    'depth': 9,
                    'learning_rate': 0.06,
                    'l2_leaf_reg': 5,
                    'random_strength': 1.2,
                    'bootstrap_type': 'Bernoulli',
                    'subsample': 0.75
                }
            },
            'attr_6': {  
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 7,
                    'learning_rate': 0.08,
                    'l2_leaf_reg': 3,
                    'random_strength': 0.7
                }
            },
            'attr_7': {  
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 6,
                    'learning_rate': 0.1,
                    'l2_leaf_reg': 2,
                    'random_strength': 0.5
                }
            },
            'attr_8': {  
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 5,
                    'learning_rate': 0.1,
                    'l2_leaf_reg': 2,
                    'random_strength': 0.3
                }
            }
        }