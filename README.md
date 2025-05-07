# Brain-Tumor-MRI-Classification-with-TensorFlow
This project is a deep learning-based image classification system designed to classify brain tumors using MRI images. It uses TensorFlow and Keras to build a convolutional neural network (CNN) capable of identifying four types of brain tumor conditions.

## 💡 Features

- Uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle via `kagglehub`
- CNN architecture built with `Keras`
- Data augmentation and normalization applied
- Implements class weighting to address class imbalance
- Evaluates performance using a confusion matrix and classification report
- Predicts tumor type from new MRI images

## 🧠 Tumor Classes

1. Glioma
2. Meningioma
3. Pituitary
4. No Tumor

## 📁 Dataset Structure

The dataset is automatically downloaded from Kaggle and split into:
- `Training/`
- `Testing/`

## 🧪 Model Architecture

- Data Augmentation (random flip, rotation, zoom)
- Rescaling layer
- 3 convolutional layers
- Dropout
- Fully connected dense layers

Optimizer: Adam with learning rate scheduling  
Loss: Sparse Categorical Crossentropy

## 🖼️ Example Predictions

The model predicts the class of an MRI image with confidence scores for each of the four categories.

```python
This image most likely belongs to Pituitary with a 98.21 percent confidence.
