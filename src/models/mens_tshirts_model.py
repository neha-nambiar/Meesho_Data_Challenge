from src.models.base_model import BaseFashionModel


class Men_Tshirts_Model(BaseFashionModel):
    def __init__(self):
        """
        Initializes the Men T-shirts model with specific attribute configurations.
        """
        super().__init__(num_attributes=5)
        self.attr_configs = {
            'attr_1': {
                'balance_strategy': 'smote',
                'model_params': {
                    'depth': 8,
                    'learning_rate': 0.08
                }
            },
            'attr_2': {
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 6,
                    'learning_rate': 0.1
                }
            },
            'attr_3': {
                'balance_strategy': 'class_weight',
                'model_params': {
                    'depth': 6,
                    'learning_rate': 0.1
                }
            },
            'attr_4': {
                'balance_strategy': 'auto',
                'model_params': {
                    'depth': 7,
                    'learning_rate': 0.09
                }
            },
            'attr_5': {
                'balance_strategy': 'smote',
                'model_params': {
                    'depth': 6,
                    'learning_rate': 0.1
                }
            }
        }
