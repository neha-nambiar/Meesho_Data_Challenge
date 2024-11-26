import gc
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from models.mens_tshirts_model import Men_Tshirts_Model
from models.sarees_model import Sarees_Model
from models.kurtis_model import Kurtis_Model
from models.womens_tshirts_model import Women_Tshirts_Model
from models.womens_tops_model import Women_Tops_Model



class UnifiedFashionModelPipeline:
    def __init__(self):
        """
        Initialize the pipeline with category-specific models.
        """
        self.models = {
            'Men Tshirts': Men_Tshirts_Model(),
            'Sarees': Sarees_Model(), 
            'Kurtis': Kurtis_Model(),
            'Women Tshirts': Women_Tshirts_Model(),
            'Women Tops & Tunics': Women_Tops_Model()
        }
        self.label_encoders = {}

    def preprocess_data(self, df: pd.DataFrame, category: str, num_attributes: int) -> pd.DataFrame:
        """
        Preprocess data for a specific category, including label encoding.

        Parameters:
        - df: Input dataframe.
        - category: Category of the fashion item.
        - num_attributes: Number of attributes for the category.

        Returns:
        - Preprocessed dataframe with encoded labels.
        """
        label_encoders = {}
        for i in range(1, num_attributes + 1):
            attr = f'attr_{i}'
            le = LabelEncoder()
            df[attr] = le.fit_transform(df[attr])
            label_encoders[attr] = le
        self.label_encoders[category] = label_encoders
        return df

    def train_model(self, category: str, train_data: pd.DataFrame, val_data: pd.DataFrame, epochs: int = 2000):
        """
        Train the model for a specific category.

        Parameters:
        - category: Category of the fashion item.
        - train_data: Training dataset.
        - val_data: Validation dataset.
        - epochs: Number of epochs to train the model.
        """
        model = self.models[category]
        print(f"Training model for category: {category}")
        model.train(train_data, val_data, epochs)
        
        torch.cuda.empty_cache()
        gc.collect()
        print(f"Finished training model for {category}")
        print()

    def predict(self, test_data: pd.DataFrame, category: str) -> pd.DataFrame:
        """
        Predict attributes for a specific category.

        Parameters:
        - test_data: Test dataset containing the features.
        - category: Category of the fashion item.

        Returns:
        - DataFrame with predictions (decoded labels).
        """
        model = self.models[category]
        print(f"Predicting for category: {category}")
        predictions = model.predict(test_data)
        label_encoders = self.label_encoders[category]
        for col in predictions.columns:
            predictions[col] = label_encoders[col].inverse_transform(predictions[col])
        return predictions

    def fill_predictions(self, test_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill predictions in the test dataframe using model predictions.

        Parameters:
        - test_df: Test dataframe

        Returns:
        - Test dataframe with predictions filled.
        """
        result_df = test_df.copy()
        for category, model in self.models.items():
            category_data = test_df[test_df['Category'] == category]
            if not category_data.empty:
                predictions = self.predict(category_data, category)
                predictions = predictions.set_index(category_data.index)
                result_df.update(predictions)
        return result_df
