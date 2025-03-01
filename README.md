# MONAI-Based 3D UNet for Medical Image Segmentation

## Overview
This project implements a **3D U-Net** using the **MONAI (Medical Open Network for AI)** framework for medical image segmentation. The model is trained on volumetric imaging datasets and can be used for tasks such as tumor segmentation, organ segmentation, or other 3D medical image processing applications.

## Features
- **3D U-Net Architecture**: Implements a deep learning-based segmentation model tailored for volumetric data.
- **MONAI Framework**: Utilizes MONAI's robust medical imaging tools and PyTorch for model training and evaluation.
- **Preprocessing and Augmentation**: Includes standard medical imaging preprocessing techniques such as resampling, normalization, and augmentation.
- **Efficient Training Pipeline**: Supports multi-GPU training and mixed-precision computation.
- **Evaluation Metrics**: Includes Dice Score, Hausdorff Distance, and other segmentation performance metrics.

## Installation
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.8+
- PyTorch
- MONAI
- NumPy
- SimpleITK
- Matplotlib
- TorchIO (optional, for additional augmentations)

You can install the required dependencies using:
```bash
pip install monai torch torchvision numpy matplotlib SimpleITK torchio
```

## Dataset
The model requires **3D medical imaging datasets** (e.g., CT or MRI scans). The dataset should be in **NIfTI (.nii, .nii.gz)** or **DICOM** format. Ensure your dataset is structured as follows:
```
/data
  ├── imagesTr  # Training images
  ├── labelsTr  # Training labels
  ├── imagesTs  # Testing images
  ├── labelsTs  # Testing labels
```

## Training
Run the training script with:
```bash
python train.py --data_dir /path/to/data --epochs 100 --batch_size 4 --lr 0.0001
```
### Training Options:
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 4)
- `--lr`: Learning rate (default: 0.0001)
- `--device`: Set to `cuda` for GPU training

## Evaluation
To evaluate the trained model:
```bash
python evaluate.py --data_dir /path/to/data --model_path /path/to/model.pth
```

## Results

![image](https://github.com/user-attachments/assets/e762d726-7aec-43cb-8c4e-738b0674e526)

This will compute Dice scores and other relevant segmentation metrics.

## Inference
To run inference on new medical images:
```bash
python inference.py --input_image /path/to/image.nii.gz --model_path /path/to/model.pth --output /path/to/output.nii.gz
```

## Results
After training, the model outputs segmentation masks which can be visualized using **ITK-SNAP** or **3D Slicer**.

## Future Improvements
- Integrating more advanced architectures (e.g., Swin UNet, nnUNet)
- Implementing semi-supervised and self-supervised learning techniques
- Expanding support for different medical imaging modalities

## References
- [MONAI Documentation](https://monai.io/)
- [U-Net Paper](https://arxiv.org/abs/1505.04597)

---
Feel free to contribute and open issues for improvements!

