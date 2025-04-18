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
from .base_model import BaseFashionModel
from .mens_tshirts_model import Men_Tshirts_Model
from .sarees_model import Sarees_Model
from .kurtis_model import Kurtis_Model
from .womens_tshirts_model import Women_Tshirts_Model
from .womens_tops_model import Women_Tops_Model

# Configuring warnings and logging
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Setting random seeds for reproducibility
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
