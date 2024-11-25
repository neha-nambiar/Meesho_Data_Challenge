# Fashion Attribute Prediction - Kaggle Competition sponsored by Meesho 

## Competition Overview
- **Sponsor**: Meesho
- **Goal**: Predict product attributes from fashion images
- **Challenge**: Accurately classify attributes like color, pattern, and sleeve length using only product images
- **Final Ranking**: 27th on the leaderboard

## Dataset Details
- **Categories**: 5 fashion categories
- **Data Files**:
  - `category_attributes.parquet`: Contains attribute names for each category
  - `train.csv`: Training data with product IDs, categories, and attributes
  - `test.csv`: Test data with product IDs and categories
  - `sample_submission.csv`: Submission template
- **Image Data**: Images stored as `{product_id}.jpg` in the `images/` folder

## 🚀 Solution Highlights

### Feature Extraction
- Multi-model approach using:
  - ResNet101V2
  - EfficientNetB0
  - ConvNeXt Base

### Preprocessing Techniques
- Visual similarity-based imputation
- Advanced feature scaling
- Handling class imbalance with SMOTE

### Model Architecture
- Category-specific models
- CatBoost classification
- Adaptive training strategies

## 🔧 Setup

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fashion-attribute-prediction.git
cd fashion-attribute-prediction
```

2. Set up a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -e .
```

### Downloading Kaggle Dataset

#### Method 1: Using Kaggle API (Recommended)

1. Install Kaggle API:
```bash
pip install kaggle
```

2. Set up Kaggle API credentials:
   - Go to your Kaggle account settings
   - Create a new API token
   - Save the `kaggle.json` file to `~/.kaggle/kaggle.json` (Linux/Mac) or `C:\Users\<Windows-username>\.kaggle\kaggle.json` (Windows)
   - Set proper file permissions: `chmod 600 ~/.kaggle/kaggle.json`

3. Download the competition dataset:
```bash
kaggle competitions download -c visual-taxonomy
unzip visual-taxonomy.zip -d data/
```

#### Method 2: Manual Download

1. Visit the [Kaggle Competition Page](https://www.kaggle.com/competitions/visual-taxonomy/)
2. Log in to Kaggle
3. Go to the "Data" tab
4. Download all competition data files
5. Create a `data/` directory in the project root
6. Extract downloaded files into the `data/` directory

## 🧠 Project Structure
```
fashion-attribute-prediction/
│
├── src/
│   ├── __init__.py
│   ├── feature_extraction.py
│   ├── data_preprocessing.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py
│   │   ├── mens_tshirts_model.py
│   │   ├── sarees_model.py
│   │   ├── kurtis_model.py
│   │   ├── womens_tshirts_model.py
│   │   └── womens_tops_model.py
│   └── pipeline.py
│
├── setup.py
└── README.md
```

### Training the Model

```bash
python -m src.pipeline
```

### Making Predictions

```bash
python -m src.pipeline --mode predict
```

## 🏆 Acknowledgements
- Meesho for sponsoring the competition
- Kaggle for hosting the challenge
- Open-source libraries: PyTorch, TensorFlow, CatBoost
