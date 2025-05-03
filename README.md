
# Copy Move Forgery Detection using Optimized CNN

This project implements a Convolutional Neural Network (CNN) to classify images from the MISD dataset using TensorFlow/Keras. It also incorporates hyperparameter tuning with Keras Tuner to optimize model performance.

## 📁 Dataset

The dataset used is called **MISD**, which contains multiple categories of image data such as:

- Au_ani
- Au_art
- Au_com
- ... (and other classes)

Ensure you have a ZIP file of the dataset named `MISD.zip` structured as follows:

```

/MISD/
├── train/
│   ├── class1/
│   └── class2/
├── validation/
│   ├── class1/
│   └── class2/
└── test/
├── class1/
└── class2/

````

## 🚀 Features

- Image classification using CNN
- Hyperparameter tuning with Keras Tuner
- Evaluation on validation/test sets
- Option to extend for data augmentation or transfer learning

## 🛠️ Installation

Clone the repository and install required dependencies:

```bash
git clone https://github.com/yourusername/misd-image-classification.git
cd misd-image-classification
pip install -r requirements.txt
````

In Google Colab, you can also install Keras Tuner directly:

```bash
!pip install keras-tuner
```

## 📌 Usage

1. Upload your dataset (`MISD.zip`) to the Colab environment.
2. Extract it using:

   ```bash
   !unzip -o /content/MISD.zip -d /content/MISD
   ```
3. Load and preprocess the data using `ImageDataGenerator` or `tf.data`.
4. Build and compile the CNN model.
5. Use Keras Tuner for hyperparameter tuning.
6. Train and evaluate the model.

## 📊 Results

The best model achieved approximately **68% accuracy** on the test dataset after tuning. (Update this after running experiments.)

### Test Accuracy on MISD dataset: 0.6811

### Classification Report for MISD dataset:

```
              precision    recall  f1-score   support

           0       0.69      0.96      0.80       124
           1       0.58      0.11      0.19        61

    accuracy                           0.68       185
   macro avg       0.64      0.54      0.50       185
weighted avg       0.65      0.
```
