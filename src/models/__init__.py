"""
Main initialization file for the fashion attribute prediction package.

This module provides initialization and imports for the feature extraction,
preprocessing, and modeling components of the fashion attribute prediction pipeline.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import tensorflow as tf

# Key modules to expose
from src.feature_extraction import (
    FeatureExtractor, 
    UnifiedFeatureExtractor
)
from src.data_preprocessing import Imputer
from src.pipeline import UnifiedFashionModelPipeline
from src.models import (
    BaseFashionModel, 
    Men_Tshirts_Model, 
    Sarees_Model, 
    Kurtis_Model,
    Women_Tshirts_Model, 
    Women_Tops_Model
)

# Configure warnings and logging
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Set random seeds for reproducibility
def set_seed(seed=42):
    """
    Set random seeds for reproducibility across libraries.
    
    Args:
        seed (int): Seed value for random number generation.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# Initialize seed on import
set_seed()

# Expose package version
__version__ = "0.1.0"