# ConText-E: Multi-Modal Sentiment Analysis for Climate Change Discourse

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accuracy](https://img.shields.io/badge/Accuracy-93.69%25-green.svg)]()

## 🎯 Overview

ConText-E is an advanced sentiment analysis system that combines **semantic**, **emotional**, and **topical** features to achieve state-of-the-art performance on climate change discourse analysis. The system uses a novel three-phase pipeline that fine-tunes DeBERTa for semantic understanding, extracts emotional features using RoBERTa, and applies BERTopic for topic modeling, all integrated through a LightGBM meta-learner.

## 🚀 Key Features

- **Multi-Modal Architecture**: Combines semantic, emotional, and topical features
- **State-of-the-Art Performance**: 93.69% accuracy with 93.67% weighted F1-score
- **Robust Validation**: 5-fold stratified cross-validation
- **Large-Scale Dataset**: ~51K+ climate change tweets
- **Reproducible Research**: Complete pipeline with fixed random seeds
- **Production Ready**: End-to-end prediction system

## 📊 Performance Results

| Metric | Score |
|--------|-------|
| **Cross-Validation Accuracy** | **93.69%** |
| **Weighted F1-Score** | **93.67%** |
| **Individual Fold Accuracy** | 93.14% - 94.12% |
| **Dataset Size** | 51,419 tweets |
| **Feature Dimensions** | 902 |

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Phase 1:      │    │   Phase 2:      │    │   Phase 3:      │
│   Feature       │    │   DeBERTa       │    │   Meta-Learner  │
│   Engineering   │    │   Fine-tuning   │    │   Training      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ • Emotional     │    │ • Semantic      │    │ • LightGBM      │
│   Features      │    │   Embeddings    │    │   Meta-Learner  │
│   (RoBERTa)     │    │   (DeBERTa)     │    │ • 5-Fold CV     │
│ • Topic         │    │ • Fine-tuned    │    │ • Feature       │
│   Features      │    │   Model         │    │   Fusion        │
│   (BERTopic)    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ConText-E_Project.git
cd ConText-E_Project
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Feature Engineering
```bash
python src/01_feature_engineering.py
```
- Generates emotional features using RoBERTa
- Creates topic features using BERTopic
- Processes ~51K tweets (15-30 minutes)

### 2. DeBERTa Fine-tuning
```bash
python src/02_finetune_deberta.py
```
- Fine-tunes DeBERTa-v3-base on sentiment data
- Saves semantic core model (1.5-3 hours)

### 3. Meta-Learner Training
```bash
python src/03_train_meta_learner.py
```
- Trains LightGBM meta-learner with 5-fold CV
- Combines all feature modalities
- Achieves 93.69% accuracy

### 4. Make Predictions
```bash
python src/predict.py
```
- Interactive sentiment prediction
- Real-time analysis of user input

## 📁 Project Structure

```
ConText-E_Project/
├── src/
│   ├── 01_feature_engineering.py    # Phase 1: Feature extraction
│   ├── 02_finetune_deberta.py       # Phase 2: DeBERTa fine-tuning
│   ├── 03_train_meta_learner.py     # Phase 3: Meta-learner training
│   └── predict.py                   # Prediction interface
├── data/
│   ├── twitter_sentiment_data.csv   # Raw dataset
│   └── enriched_data.csv            # Processed features
├── saved_models/
│   ├── bertopic_model/              # Topic modeling model
│   ├── deberta_semantic_core/       # Fine-tuned DeBERTa
│   └── lgbm_meta_learner.pkl        # Meta-learner model
├── results/                         # Training checkpoints
└── requirements.txt                 # Dependencies
```

## 🔬 Research Contributions

### Novel Methodology
- **Multi-Modal Feature Fusion**: First to combine semantic, emotional, and topical features for sentiment analysis
- **Embedding-Based Meta-Learning**: Uses transformer embeddings as features rather than direct classification
- **Robust Validation**: Comprehensive cross-validation with proper stratification

### Technical Innovations
- **Semantic Features**: DeBERTa-v3-base embeddings (768 dimensions)
- **Emotional Features**: RoBERTa emotion classification (28 emotion categories)
- **Topical Features**: BERTopic clustering with one-hot encoding
- **Meta-Learning**: LightGBM ensemble for final prediction

## 📈 Experimental Results

### Cross-Validation Performance
| Fold | Accuracy | F1-Score |
|------|----------|----------|
| 1    | 93.14%   | 93.12%   |
| 2    | 94.03%   | 94.01%   |
| 3    | 93.52%   | 93.50%   |
| 4    | 94.12%   | 94.10%   |
| 5    | 93.64%   | 93.62%   |
| **Average** | **93.69%** | **93.67%** |

### Feature Analysis
- **Total Features**: 902 dimensions
- **Semantic Features**: 768 (DeBERTa embeddings)
- **Emotional Features**: 28 (emotion categories)
- **Topical Features**: 106 (topic clusters)

## 🎯 Use Cases

- **Climate Change Research**: Analyze public sentiment on climate discourse
- **Social Media Monitoring**: Track sentiment trends on environmental topics
- **Policy Analysis**: Understand public opinion on climate policies
- **Academic Research**: Study sentiment patterns in climate communication

## 🔧 Configuration

### Model Parameters
```python
# DeBERTa Fine-tuning
MODEL_NAME = "microsoft/deberta-v3-base"
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
EPOCHS = 3

# LightGBM Meta-Learner
N_ESTIMATORS = 500
CV_FOLDS = 5
RANDOM_STATE = 42
```

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ free space

## 📚 Dependencies

### Core Libraries
- `transformers==4.41.2` - Hugging Face transformers
- `torch` - PyTorch deep learning framework
- `lightgbm` - Gradient boosting meta-learner
- `bertopic` - Topic modeling
- `scikit-learn` - Machine learning utilities

### Data Processing
- `pandas` - Data manipulation
- `numpy<2.0` - Numerical computing
- `datasets==2.19.0` - Dataset handling

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/yourusername/ConText-E_Project.git
cd ConText-E_Project
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Development dependencies
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📖 Citation

If you use this work in your research, please cite:

```bibtex
@article{context_e_2024,
  title={ConText-E: Multi-Modal Sentiment Analysis for Climate Change Discourse},
  author={Your Name},
  journal={Conference/Journal Name},
  year={2024},
  url={https://github.com/yourusername/ConText-E_Project}
}
```

## 🙏 Acknowledgments

- **Hugging Face** for transformer models and libraries
- **Microsoft** for DeBERTa-v3-base model
- **LightGBM** team for the gradient boosting framework
- **BERTopic** for topic modeling capabilities

## 📞 Contact

- **Author**: Your Name
- **Email**: your.email@university.edu
- **GitHub**: [@yourusername](https://github.com/yourusername)
- **LinkedIn**: [Your Profile](https://linkedin.com/in/yourprofile)

## 🔮 Future Work

- [ ] Cross-domain evaluation on other sentiment datasets
- [ ] Ablation studies for feature importance
- [ ] Real-time streaming sentiment analysis
- [ ] Multi-language support
- [ ] Interpretability analysis

---

**⭐ If you find this project useful, please give it a star!**
