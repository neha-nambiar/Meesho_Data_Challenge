import os
import argparse
import pandas as pd
import gc
import torch
from src.pipeline import UnifiedFashionModelPipeline
from download_kaggle_data import download_data


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train and predict clothing product attributes.")
    parser.add_argument('--download_data', action='store_true', help="Download dataset from Kaggle.")
    parser.add_argument('--train', action='store_true', help="Train models on the dataset.")
    parser.add_argument('--predict', action='store_true', help="Make predictions on the test set.")
    args = parser.parse_args()

    # Paths
    data_dir = "data"
    train_images_dir = os.path.join(data_dir, "train_images")
    test_images_dir = os.path.join(data_dir, "test_images")
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    
    if args.download_data:
        print("Downloading dataset from Kaggle...")
        download_data()  # Assumes the dataset download is handled in this script
        print("Dataset downloaded successfully.")

    if args.train:
        print("Starting training process...")
        # Load training data
        print("Loading training data...")
        data = pd.read_csv(train_csv)
        data['image_path'] = data['id'].apply(lambda x: os.path.join(train_images_dir, f"{str(x).zfill(6)}.jpg"))
        data = data.drop(['id'], axis=1)

        # Split data by category
        datasets = {
            'Men_Tshirts': data[data['Category'] == 'Men Tshirts'].reset_index(drop=True),
            'Sarees': data[data['Category'] == 'Sarees'].reset_index(drop=True),
            'Kurtis': data[data['Category'] == 'Kurtis'].reset_index(drop=True),
            'Women_Tshirts': data[data['Category'] == 'Women Tshirts'].reset_index(drop=True),
            'Womens_Tops': data[data['Category'] == 'Women Tops & Tunics'].reset_index(drop=True)
        }

        # Initialize the pipeline
        pipeline = UnifiedFashionModelPipeline()

        # Train models for each category
        for category, dataset_df in datasets.items():
            print(f"\nTraining model for category: {category}")
            pipeline.train_model(category, dataset_df)

        print("Training completed successfully.")

    if args.predict:
        print("Starting prediction process...")
        # Load test data
        print("Loading test data...")
        test_data = pd.read_csv(test_csv)
        test_data['image_path'] = test_data['id'].apply(lambda x: os.path.join(test_images_dir, f"{str(x).zfill(6)}.jpg"))

        # Initialize the pipeline
        pipeline = UnifiedFashionModelPipeline()

        # Make predictions
        print("Generating predictions...")
        predictions = pipeline.fill_predictions(test_data)

        # Save predictions
        output_file = os.path.join(data_dir, "predictions.csv")
        predictions.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    # Release GPU memory and garbage collect after each step
    try:
        main()
    finally:
        torch.cuda.empty_cache()
        gc.collect()
