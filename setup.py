from setuptools import setup, find_packages

setup(
    name='Meesho_Data_Challenge',
    author='Neha Mahendran Nambiar',
    description='Kaggle competition code',
    url='https://github.com/neha-nambiar/Meesho_Data_Challenge',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    
    # Dependencies
    install_requires=[
        # Core data science and ML libraries
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        
        # Deep learning frameworks
        'torch>=1.10.0',
        'torchvision>=0.11.0',
        'tensorflow>=2.7.0',
        
        # Image processing
        'Pillow>=8.4.0',
        'opencv-python-headless>=4.5.0',
        
        # Feature extraction and model libraries
        'timm>=0.5.0',
        'catboost>=1.1.0',
        
        # Additional utilities
        'tqdm>=4.62.0',
        'scipy>=1.7.0',
        'imbalanced-learn>=0.9.0'
    ],
    
    # Optional dependencies
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'jupyter>=1.0.0',
            'matplotlib>=3.4.0',
            'seaborn>=0.11.0'
        ],
        'gpu': [
            'cuda-python>=11.0.0',
            'nvidia-cudnn-cu11>=8.0.0'
        ]
    },
    
    # Additional project information
    keywords='meesho-data-challenge machine-learning computer-vision',
    
    # Python version requirements
    python_requires='>=3.8,<3.11',
    
    # Entry points (if you want to create CLI commands)
    entry_points={
        'console_scripts': [
            'fashion-predict=src.pipeline:main',
        ],
    },
    
    # Include package data
    include_package_data=True,
    package_data={
        '': ['*.yml', '*.yaml', '*.json', '*.txt'],
    },
)
