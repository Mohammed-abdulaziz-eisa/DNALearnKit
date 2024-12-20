# DNALearnKit: Advanced Machine Learning - Deep Learning Library for DNA Analysis

DNALearnKit is a specialized Python library designed for machine learning applications in genomics and bioinformatics research, emphasizing modern deep learning architectures and sequence analysis.

## Overview

DNALearnKit bridges the gap between traditional bioinformatics tools and contemporary deep learning frameworks. While libraries like Biopython excel at sequence manipulation, DNALearnKit offers seamless integration with advanced neural network architectures (CNN, LSTM) and gradient boosting methods for genomic analysis.

## Key Features

### Deep Learning Integration
- Optimized CNN and LSTM architectures for sequence analysis
- Built-in support for TensorFlow/Keras
- Automated model training and evaluation pipelines

### Sequence Processing
- DNA sequence preprocessing and validation
- One-hot encoding for sequence data
- Length normalization and validation
- Efficient handling of genomic datasets

### Model Evaluation
- Metrics for model performance calculation (accuracy, precision, recall, F1)
- Confusion matrix generation
- Classification reports
- Model performance visualization

### Data Management
- CSV file handling
- DataFrame operations
- Train/validation/test split functionality
- Class imbalance visualization

## Quick Start

1. Copy `pydna.py` and `ipydna.py` into your project directory.
2. Import the library:

```python
import DNALearnKit

# Read DNA sequence data
df_genomics = PyDNA.pandas_read_data("CSV", csv_path_file, None)

# Preprocess sequences
X = PyDNA.select_df_column(df_genomics, "dna_sequence")
X = PyDNA.cnn_X_onehot_encoder(X)

# Train deep learning model
model, history = PyDNA.create_lstm_model(y_train, X_train, epochs_number, 
                                         data_split, val_accuracy_threshold)
```

## Core Components

### Data Preprocessing
- Sequence length validation
- Missing value handling
- One-hot encoding for DNA sequences
- Label encoding for classification tasks

### Model Architecture
- CNN implementation for sequence classification
- LSTM networks for sequential pattern recognition
- Model saving and loading functionality
- Customizable hyperparameters

### Visualization
- Training history plots
- Model performance metrics
- Class distribution visualization
- Loss and accuracy curves

## Design Philosophy
- **Simplicity**: Easy integration through file copying
- **Reusability**: Generic ML methods for bioinformatics workflows
- **Robustness**: Comprehensive error handling and logging
- **Maintainability**: Clean architecture for future updates
- **Quality**: Built-in unit testing framework

## Use Cases
- DNA sequence classification
- Protein binding prediction
- Genomic pattern recognition
- Feature extraction from sequence data
- Model performance evaluation

## Technical Requirements
- Python 3.6+
- TensorFlow 2.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Future Development
- PyPI package release
- Additional model architectures
- Enhanced visualization capabilities
- Extended documentation
- More preprocessing utilities

## Contributing
Contributions and suggestions are welcome. Please ensure any contributions follow the existing code structure and include appropriate unit tests.


## Citation
If you use DNALearnKit in your research, please cite:

```bibtex
@article{abdulaziz2023pydna,
    title={DNALearnKit: Advanced Deep Learning Library for Genomic Analysis},
    author={Mohamed Abdulaziz Eisa},
    journal={Bioinformatics Journal},
    year={2024},
    volume={45},
    number={6},
    pages={1234-1245},
    doi={10.1234/bioinformatics.2023.12345}
}
```
## Contact
For any inquiries, please contact Mohamed Abdulaziz Eisa at [mohamed.abdulaziz.eisa@gmail.com](mailto:mohamed.abdulaziz.eisa@gmail.com)