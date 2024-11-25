from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fashion-attribute-prediction',
    version='0.1.0',
    author='Neha Mahendran Nambiar',
    description='A comprehensive fashion attribute prediction model',
    long_description=long_description,
    long_description_content_type='text/markdown',
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
    
    # Metadata for PyPI
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    
    # Additional project information
    keywords='fashion-attribute-prediction machine-learning computer-vision',
    
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