from src.models.base_model import BaseFashionModel

class Sarees_Model(BaseFashionModel):
    def __init__(self):
        """
        Initializes the Sarees model with specific attribute configurations
        and preprocessing steps.
        """
        super().__init__(num_attributes=10)
        self.attr_configs = {
            'attr_1': {  
                'balance_strategy': 'hybrid',
                'model_params': {
                    'depth': 8,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 3,
                    'min_data_in_leaf': 10,
                    'random_strength': 1
                }
            },
            'attr_2': {  
                'balance_strategy': 'hybrid',
                'model_params': {
                    'depth': 8,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 5,
                    'min_data_in_leaf': 15
                }
            },
            'attr_3': {  
                'balance_strategy': 'smote',
                'model_params': {
                    'depth': 7,
                    'learning_rate': 0.08,
                    'l2_leaf_reg': 3,
                    'min_data_in_leaf': 10
                }
            },
            'attr_4': {  
                'balance_strategy': 'hybrid',
                'model_params': {
                    'depth': 9,
                    'learning_rate': 0.03,
                    'l2_leaf_reg': 7,
                    'min_data_in_leaf': 20
                }
            },
            'attr_5': { 
                'balance_strategy': 'hybrid',
                'model_params': {
                    'depth': 8,
                    'learning_rate': 0.04,
                    'l2_leaf_reg': 5,
                    'min_data_in_leaf': 15,
                    'random_strength': 1
                }
            },
            'attr_6': {  
                'balance_strategy': 'smote',
                'model_params': {
                    'depth': 7,
                    'learning_rate': 0.08,
                    'l2_leaf_reg': 3,
                    'min_data_in_leaf': 10
                }
            },
            'attr_7': {  
                'balance_strategy': 'hybrid',
                'model_params': {
                    'depth': 8,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 5,
                    'min_data_in_leaf': 15
                }
            },
            'attr_8': {  
                'balance_strategy': 'hybrid',
                'model_params': {
                    'depth': 8,
                    'learning_rate': 0.05,
                    'l2_leaf_reg': 5,
                    'min_data_in_leaf': 15
                }
            },
            'attr_9': {  # print (9 classes)
                'balance_strategy': 'hybrid',
                'model_params': {
                    'depth': 9,
                    'learning_rate': 0.03,
                    'l2_leaf_reg': 7,
                    'min_data_in_leaf': 15,
                    'random_strength': 1
                }
            },
            'attr_10': {  
                'balance_strategy': 'smote',
                'model_params': {
                    'depth': 7,
                    'learning_rate': 0.08,
                    'l2_leaf_reg': 3,
                    'min_data_in_leaf': 10
                }
            }
        }