import gc
import timm
import torch
import tensorflow as tf
from typing import List, Tuple
from PIL import Image
import warnings
import numpy as np
from tqdm import tqdm



class FeatureExtractor:
    def __init__(self, model_name: str, device: str = None):
        """
        Initializes the feature extractor with a specified timm model.

        Parameters:
        - model_name: Name of the model (compatible with timm).
        - device: Device for computation (e.g., 'cuda' or 'cpu').
        """
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = timm.create_model(model_name, pretrained=True, num_classes=0)  
        self.model.to(self.device)
        self.model.eval()

        data_config = timm.data.resolve_model_data_config(self.model)
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

        if self.device.type == "cuda":
            self.model = self.model.half()  

    def preprocess_batch(self, image_paths: List[str]) -> Tuple[torch.Tensor, List[int]]:
        """
        Preprocess a batch of image paths into tensors for the model.

        Parameters:
        - image_paths: List of image file paths.

        Returns:
        - Tuple of batched tensor and a list of valid indices corresponding to the processed images.
        """
        valid_images = []
        valid_indices = []

        for idx, path in enumerate(image_paths):
            try:
                image = Image.open(path).convert("RGB")
                transformed_image = self.transforms(image)
                valid_images.append(transformed_image)
                valid_indices.append(idx)
            except Exception as e:
                warnings.warn(f"Error loading image {path}: {str(e)}")

        if not valid_images:
            return None, []

        batch_tensor = torch.stack(valid_images)
        batch_tensor = batch_tensor.to(self.device, dtype=torch.half if self.device.type == "cuda" else torch.float)
        return batch_tensor, valid_indices

    def process_batch(self, batch_tensor: torch.Tensor) -> np.ndarray:
        """
        Pass a preprocessed batch of images through the model to extract features.

        Parameters:
        - batch_tensor: A batch of preprocessed image tensors.

        Returns:
        - Numpy array of extracted features.
        """
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
            features = self.model.forward_features(batch_tensor)
            features = self.model.forward_head(features, pre_logits=True)
        return features.cpu().numpy()

    def process_dataset(self, df, batch_size=8): 
        features_list = []
        valid_indices = []
    
        for i in tqdm(range(0, len(df), batch_size), desc="Processing images"):
            batch_paths = df['image_path'].iloc[i:i + batch_size].tolist()
            batch_tensor, batch_valid_indices = self.preprocess_batch(batch_paths)
    
            if batch_tensor is None or not batch_valid_indices:
                continue
    
            batch_features = self.process_batch(batch_tensor)
            features_list.append(batch_features)
            valid_indices.extend([i + idx for idx in batch_valid_indices])
    
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
    
        all_features = np.vstack(features_list) if features_list else np.array([])
        processed_df = df.iloc[valid_indices].copy()
        return processed_df, all_features

    def __del__(self):
        """
        Destructor to clean up resources.
        """
        try:
            del self.model
            del self.transforms
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        except:
            pass
        
        
        
class UnifiedFeatureExtractor:
    def __init__(self):
        """
        Initialize feature extractors for all models.
        """
        self.keras_extractors = {
            'resnet': self.build_resnet_feature_extractor(),
            'efficientnet': self.build_efficientnet_feature_extractor(),
            'convnext': self.build_convnext_feature_extractor()
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build_resnet_feature_extractor(self):
        base_model = tf.keras.applications.ResNet101V2(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet"
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        return tf.keras.Model(inputs=base_model.input, outputs=x)

    def build_efficientnet_feature_extractor(self):
        base_model = tf.keras.applications.EfficientNetB0(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet"
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        return tf.keras.Model(inputs=base_model.input, outputs=x)

    def build_convnext_feature_extractor(self):
        base_model = tf.keras.applications.ConvNeXtBase(
            input_shape=(224, 224, 3), include_top=False, weights="imagenet"
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        return tf.keras.Model(inputs=base_model.input, outputs=x)

    def extract_features(self, df, model_name):
        """
        Extract features using models ResNet, EfficientNet and ConvNeXt.
        """
        model = self.keras_extractors[model_name]
        dataset = self.create_tf_dataset(df['image_path'].values, model_name)
        features = model.predict(dataset, verbose=1)
        column_name = f"image_features_{model_name}"
        df[column_name] = list(features)
        return df

    def create_tf_dataset(self, image_paths, model_name):
        """
        Create a TensorFlow dataset for feature extraction.
        """
        def preprocess_image(image_path):
            img_str = tf.io.read_file(image_path)
            img = tf.image.decode_jpeg(img_str, channels=3)
            img = tf.image.resize(img, [224, 224])
            if model_name == 'resnet':
                img = tf.keras.applications.resnet_v2.preprocess_input(img)
            elif model_name == 'efficientnet':
                img = tf.keras.applications.efficientnet.preprocess_input(img)
            elif model_name == 'convnext':
                img = tf.keras.applications.convnext.preprocess_input(img)
            return img

        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        return dataset

    def extract_features(self, df, model_names):
        """
        Extract features for a dataset using the models.
        """
        for model_name in model_names:
            print(f"Extracting features with {model_name}...")
            df = self.extract_features(df, model_name)
        return df
