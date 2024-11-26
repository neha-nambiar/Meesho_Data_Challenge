# <div align="center"> Visual Taxonomy - Kaggle Competition sponsored by Meesho </div>

## 🎯 Competition Overview
- **Sponsor**: Meesho
- **Goal**: Predict product attributes from fashion images
- **Challenge**: Accurately classify attributes like color, pattern, and sleeve length using only product images
- **Final Ranking**: 27th on the leaderboard

## 📊 Dataset Details
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
git clone https://github.com/neha-nambiar/Meesho_Data_Challenge.git
cd Meesho_Data_Challenge
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

## 🧠 Project Structure
```
Meesho_Data_Challenge/
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
