# Brain Tumor Detection using Vision Transformer (ViT)

## Project Overview
This project implements a brain tumor detection system using Vision Transformer (ViT) architecture with transfer learning. The system is designed to classify brain MRI scans into two categories: tumor-present and tumor-absent cases.

## Model Evolution
### Initial Approach: CNN
Initially, the project was implemented using a Convolutional Neural Network (CNN) architecture. However, this approach faced significant overfitting issues:
- The model showed high training accuracy but poor generalization
- Large gap between training and validation performance
- Limited ability to capture complex features in medical imaging

### Current Approach: Vision Transformer (ViT)
To address these limitations, the project was migrated to a Vision Transformer architecture using transfer learning. This approach showed significant improvements:
- Better generalization performance
- Reduced overfitting
- More stable training metrics
- Improved feature extraction capabilities

## Technical Implementation

### Dependencies
```python
torch
torchvision
numpy
opencv-python (cv2)
scikit-learn
matplotlib
```

### Dataset Structure
The dataset should be organized as follows:
```
brain_tumor_dataset/
├── yes/         # Contains MRI scans with tumors
└── no/          # Contains MRI scans without tumors
```

### Key Components
1. **Data Processing**
   - Image resizing to 224x224
   - RGB color conversion
   - Normalization
   - Data augmentation through ViT transforms

2. **Model Architecture**
   - Pretrained ViT-B/16 backbone
   - Modified classifier head for binary classification
   - Frozen base parameters for transfer learning

3. **Training Pipeline**
   - Binary Cross-Entropy Loss
   - Adam optimizer
   - Learning rate: 1e-3
   - Batch size: 32
   - 10 epochs with early stopping

## Results Visualization
The training process includes visualization of:
- Training and validation loss curves
- Accuracy metrics over time
- Performance comparison plots

## Usage

1. **Setup Environment**
```bash
pip install torch torchvision numpy opencv-python scikit-learn matplotlib
```

2. **Configure Paths**
Update the data paths in main():
```python
tumor_path = "/path/to/brain_tumor_dataset/yes"
healthy_path = "/path/to/brain_tumor_dataset/no"
```

3. **Run Training**
```bash
python brain_tumor_detection.py
```

## Model Performance
- The ViT model shows consistent performance across training and testing sets
- Reduced overfitting compared to the CNN approach
- Training and validation metrics are closely aligned
- Model checkpointing saves the best performing version

## Future Improvements
- Implementation of additional data augmentation techniques
- Experimentation with different ViT architectures
- Integration of cross-validation
- Addition of explainability techniques
- Ensemble methods with other architectures

## Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT License](https://choosealicense.com/licenses/mit/)

## Acknowledgments
- Dataset source:  https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection
- Vision Transformer implementation based on the original paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
- Transfer learning approach inspired by recent advances in medical image analysis
