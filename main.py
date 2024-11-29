import argparse
import os
import gc
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from src.pipeline import UnifiedFashionModelPipeline
from src.feature_extraction import UnifiedFeatureExtractor


def preprocess_data(data_dir, train_file, test_file, img_dir):
    """Preprocess the training and test datasets."""
    print("Preprocessing data...")
    # Load training data
    train_df = pd.read_csv(os.path.join(data_dir, train_file))
    train_df['image_path'] = train_df['id'].apply(lambda x: os.path.join(img_dir, 'train_images', f"{str(x).zfill(6)}.jpg"))
    train_df.drop(columns=['id'], inplace=True)

    # Load test data
    test_df = pd.read_csv(os.path.join(data_dir, test_file))
    test_df['image_path'] = test_df['id'].apply(lambda x: os.path.join(img_dir, 'test_images', f"{str(x).zfill(6)}.jpg"))
    return train_df, test_df


def extract_features(train_df, model_names):
    """Extract features using pre-defined models."""
    print("Extracting features...")
    feature_extractor = UnifiedFeatureExtractor()
    categories = train_df['Category'].unique()
    datasets = {
        category: train_df[train_df['Category'] == category].reset_index(drop=True)
        for category in categories
    }

    for model_name in model_names:
        for category, dataset_df in datasets.items():
            print(f"Processing {category} with {model_name}...")
            datasets[category] = feature_extractor.extract_features(dataset_df.copy(), [model_name])
            torch.cuda.empty_cache()
            gc.collect()

    return pd.concat(datasets.values(), ignore_index=True)


def train_models(train_df):
    """Train models for each category."""
    print("Training models...")
    pipeline = UnifiedFashionModelPipeline()

    for category, model in pipeline.models.items():
        print(f"Training for category: {category}")
        category_data = train_df[train_df['Category'] == category].copy()
        num_attributes = len(model.attributes)

        # Filter valid columns
        valid_columns = {f"attr_{i}" for i in range(1, num_attributes + 1)}
        columns_to_keep = [col for col in category_data.columns if not col.startswith("attr_") or col in valid_columns]
        category_data = category_data[columns_to_keep].dropna()

        # Split into training and validation sets
        train_data, val_data = train_test_split(category_data, test_size=0.2, random_state=42)
        pipeline.train_model(category, train_data, val_data)

        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()


def predict(test_df):
    """Predict on the test dataset."""
    print("Generating predictions...")
    pipeline = UnifiedFashionModelPipeline()
    predictions = pipeline.fill_predictions(test_df)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Train and predict fashion attributes.")
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True,
                        help="Mode to run the script: 'train' or 'predict'")
    parser.add_argument('--data_dir', type=str, default='data', help="Path to the data directory")
    parser.add_argument('--train_file', type=str, default='train.csv', help="Name of the training data file")
    parser.add_argument('--test_file', type=str, default='test.csv', help="Name of the test data file")
    parser.add_argument('--img_dir', type=str, default='data', help="Path to the image directory")
    parser.add_argument('--output_file', type=str, default='predictions.csv', help="File to save predictions")
    args = parser.parse_args()

    if args.mode == 'train':
        # Preprocess data
        train_df, _ = preprocess_data(args.data_dir, args.train_file, args.test_file, args.img_dir)

        # Extract features
        model_names = ['resnet', 'efficientnet', 'convnext']
        train_df = extract_features(train_df, model_names)

        # Train models
        train_models(train_df)

    elif args.mode == 'predict':
        # Preprocess test data
        _, test_df = preprocess_data(args.data_dir, args.train_file, args.test_file, args.img_dir)

        # Generate predictions
        predictions = predict(test_df)

        # Save predictions to file
        predictions.to_csv(args.output_file, index=False)
        print(f"Predictions saved to {args.output_file}")


if __name__ == "__main__":
    main()
