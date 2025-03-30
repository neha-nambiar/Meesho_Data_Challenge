# <div align="center"> Visual Taxonomy - Kaggle Competition sponsored by Meesho </div>

## 🎯 Competition Overview
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

## 🛠️ Technical Stack

- **Deep Learning Models**: ResNet101V2, EfficientNetB0, ConvNeXt Base
- **ML Framework**: CatBoost with category-specific models
- **Preprocessing**: SMOTE, SMOTETomek, Class Weight Balancing
- **Libraries**: PyTorch, TensorFlow, scikit-learn, TIMM

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


## 🧠 Project Structure
```
Meesho_Data_Challenge/
│
├── notebooks/
│   ├── data-exploration.ipynb
│   ├── model-training-demo.ipynb
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
├── download_kaggle_data.py
├── main.py
├── requirements.txt
└── README.md
```
