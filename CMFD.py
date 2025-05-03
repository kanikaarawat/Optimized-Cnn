# %%
!pip install keras-tuner


# %%


# %%
!unzip -o /content/MISD.zip -d /content/MISD


# %%

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import keras_tuner as kt

# %%
import os
import numpy as np
from PIL import Image

# Set the paths to the MISD dataset directories
authentic_folder = '/content/MISD/Dataset/Au'
spliced_folder = '/content/MISD/Dataset/Sp'


# Desired size for resizing images (adjust as needed)
img_size = (256, 256)

def load_MISD_images_and_labels(auth_folder, spliced_folder, img_size):
    images = []
    labels = []

    # Process authentic images (label=0)
    for file_name in os.listdir(auth_folder):
        # Adjust the extensions as needed. MISD authentic images are in JPEG format.
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(auth_folder, file_name)
            try:
                img = Image.open(file_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(0)  # Authentic
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    # Process spliced (forged) images (label=1)
    for file_name in os.listdir(spliced_folder):
        # Adjust the extensions as needed. MISD spliced images are in PNG format.
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(spliced_folder, file_name)
            try:
                img = Image.open(file_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(1)  # Spliced/forged
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return np.array(images), np.array(labels)

# Load images and labels from the MISD dataset
all_images, all_labels = load_MISD_images_and_labels(authentic_folder, spliced_folder, img_size)

print("Total images:", len(all_images))
print("Total labels:", len(all_labels))
print("Authentic images count:", np.sum(all_labels == 0))
print("Spliced images count:", np.sum(all_labels == 1))

# %%
import os

# Set the MISD dataset path
misd_authentic_folder = "/content/MISD/Dataset/Au"
misd_spliced_folder = "/content/MISD/Dataset/Sp"

# List all files in the authentic folder
authentic_files = sorted(os.listdir(misd_authentic_folder))
print("Total authentic images:", len(authentic_files))
print("Sample authentic files:", authentic_files[:20])  # First 20 authentic images

# List all files in the spliced folder
spliced_files = sorted(os.listdir(misd_spliced_folder))
print("\nTotal spliced images:", len(spliced_files))
print("Sample spliced files:", spliced_files[:20])  # First 20 spliced images


# %%
import os
import numpy as np
from PIL import Image

# Paths to MISD dataset folders
authentic_folder = '/content/MISD/Dataset/Au'
spliced_folder = '/content/MISD/Dataset/Sp'
img_size = (256, 256)

def load_MISD_images_and_labels(auth_folder, spliced_folder, img_size):
    images = []
    labels = []

    # Load authentic images (label 0)
    for file_name in sorted(os.listdir(auth_folder)):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
            file_path = os.path.join(auth_folder, file_name)
            img = Image.open(file_path).convert('RGB')
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(0)

    # Load spliced (forged) images (label 1)
    for file_name in sorted(os.listdir(spliced_folder)):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png', '.tif')):
            file_path = os.path.join(spliced_folder, file_name)
            img = Image.open(file_path).convert('RGB')
            img = img.resize(img_size)
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(1)

    return np.array(images), np.array(labels)

# Load images and labels
all_images, all_labels = load_MISD_images_and_labels(authentic_folder, spliced_folder, img_size)

# Display dataset stats
print("Total images loaded:", len(all_images))
print("Total authentic:", np.sum(all_labels == 0))
print("Total spliced:", np.sum(all_labels == 1))


# %%
from sklearn.model_selection import train_test_split

# Split into train+val and test (80/20), with stratification
X_train_val, X_test, y_train_val, y_test = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42, stratify=all_labels
)

# Split train+val into train and validation (75/25 of 80%, i.e., 60/20)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

# Print final split sizes
print("Train set size:", len(X_train))
print("Validation set size:", len(X_val))
print("Test set size:", len(X_test))


# %%
import os
import numpy as np
from PIL import Image

def load_MISD_images_and_labels(auth_folder, spliced_folder, img_size=(256, 256)):
    images = []
    labels = []

    # Load authentic images (label = 0)
    for file_name in sorted(os.listdir(auth_folder)):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(auth_folder, file_name)
            try:
                img = Image.open(file_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(0)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    # Load spliced (forged) images (label = 1)
    for file_name in sorted(os.listdir(spliced_folder)):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(spliced_folder, file_name)
            try:
                img = Image.open(file_path).convert('RGB')
                img = img.resize(img_size)
                img_array = np.array(img) / 255.0
                images.append(img_array)
                labels.append(1)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return np.array(images), np.array(labels)


# %%
authentic_folder = '/content/MISD/Dataset/Au'
spliced_folder = '/content/MISD/Dataset/Sp'
all_images, all_labels = load_MISD_images_and_labels(authentic_folder, spliced_folder)


# %%
from sklearn.model_selection import train_test_split

# First split: 80% train+val, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    all_images,
    all_labels,
    test_size=0.2,
    random_state=42,
    stratify=all_labels
)

# Second split: from train_val => 75% train, 25% val (which makes it 60% train, 20% val, 20% test overall)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify=y_train_val
)

# Summary
print("Train authentic:", sum(y_train==0))
print("Train forged:", sum(y_train==1))
print("Validation authentic:", sum(y_val==0))
print("Validation forged:", sum(y_val==1))
print("Test authentic:", sum(y_test==0))
print("Test forged:", sum(y_test==1))



# %%
from tensorflow import keras
from tensorflow.keras import layers, Sequential

# Use img_size = (256, 256) as defined earlier
def build_model(hp):
    model = Sequential()

    # Number of convolutional layers (between 3 and 5)
    num_conv_layers = hp.Int('num_conv_layers', min_value=3, max_value=5, step=1)

    # First Conv2D layer (with input shape)
    filters = hp.Int('filters_0', min_value=32, max_value=128, step=32)
    kernel_size = hp.Choice('kernel_size_0', values=[3, 5])
    model.add(layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
                            activation='relu', padding='same',
                            input_shape=(img_size[0], img_size[1], 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    # Optional dropout after first conv block
    dropout_rate = hp.Float('dropout_rate_0', min_value=0.0, max_value=0.5, step=0.1)
    if dropout_rate > 0:
        model.add(layers.Dropout(dropout_rate))

    # Additional convolutional layers
    for i in range(1, num_conv_layers):
        filters = hp.Int(f'filters_{i}', min_value=32, max_value=128, step=32)
        kernel_size = hp.Choice(f'kernel_size_{i}', values=[3, 5])
        model.add(layers.Conv2D(filters=filters,
                                kernel_size=(kernel_size, kernel_size),
                                activation='relu', padding='same'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        dropout_rate = hp.Float(f'dropout_rate_{i}', min_value=0.0, max_value=0.5, step=0.1)
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))

    # Flatten and dense layer
    model.add(layers.Flatten())
    model.add(layers.Dense(units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
                           activation='relu'))

    dense_dropout = hp.Float('dense_dropout', min_value=0.0, max_value=0.5, step=0.1)
    if dense_dropout > 0:
        model.add(layers.Dropout(dense_dropout))

    # Output layer for binary classification (authentic vs spliced)
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile model
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# %%
import keras_tuner as kt

tuner = kt.BayesianOptimization(
    build_model,
    objective='val_accuracy',
    max_trials=10,              # You can increase this for deeper search
    executions_per_trial=1,     # Set to >1 to average results for stability
    directory='misd_tuner_dir', # Folder to store tuning logs/results
    project_name='misd_cnn_bayes'
)


# %%
stop_early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

tuner.search(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[stop_early]
)


# %%



best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print("Best number of Conv layers:", best_hps.get('num_conv_layers'))
print("Best initial filters:", best_hps.get('filters_0'))
print("Best kernel size:", best_hps.get('kernel_size_0'))
print("Best dropout rate (first layer):", best_hps.get('dropout_rate_0'))
print("Best dense units:", best_hps.get('dense_units'))
print("Best dense dropout:", best_hps.get('dense_dropout'))
print("Best learning rate:", best_hps.get('learning_rate'))


# %%
# Build the model using the best hyperparameters from the tuner
model = tuner.hypermodel.build(best_hps)

# Train the model on the MISD dataset
history = model.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_val, y_val),
    callbacks=[stop_early]
)


# %%
# Evaluate the model on the MISD test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy on MISD dataset:", test_acc)

# Import metrics
from sklearn.metrics import classification_report, roc_auc_score

# Predict probabilities on test set
y_pred_proba = model.predict(X_test)
# Convert probabilities to binary predictions
y_pred = (y_pred_proba > 0.5).astype("int32")

# Print classification report and AUC
print("Classification Report for MISD dataset:")
print(classification_report(y_test, y_pred))
print("AUC for MISD dataset:", roc_auc_score(y_test, y_pred_proba))


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy on MISD Dataset')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss on MISD Dataset')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# %%
import random
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# Folder containing MISD images (both Au and Sp)
misd_folder = '/content/MISD/Dataset'
all_misd_images = []

# Collect image paths from both authentic and spliced folders
for subfolder in ['Au', 'Sp']:
    subfolder_path = os.path.join(misd_folder, subfolder)
    image_files = [os.path.join(subfolder_path, f) for f in os.listdir(subfolder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    all_misd_images.extend(image_files)

# Select a random image
random_image_path = random.choice(all_misd_images)

# Load the image
img = Image.open(random_image_path).convert('RGB')
img_resized = img.resize((256, 256))

# Normalize the image
img_array = np.array(img_resized) / 255.0
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Display the image
plt.imshow(img_resized)
plt.axis("off")
plt.title(f"Testing Image: {os.path.basename(random_image_path)}")
plt.show()


# %%
# Make prediction on the selected image
prediction = model.predict(img_array)[0][0]  # Single image prediction

from sklearn.metrics import precision_recall_curve
import numpy as np

# Predict probabilities on the validation set
y_pred_proba = model.predict(X_val).flatten()

# Compute precision-recall curve
precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)

# Calculate F1 scores for each threshold
f1_scores = (2 * precision * recall) / (precision + recall + 1e-8)  # Add epsilon to avoid division by zero
best_threshold = thresholds[np.argmax(f1_scores)]
print("Best Decision Threshold:", best_threshold)

# Use best threshold for final predictions
threshold = best_threshold
y_pred = (y_pred_proba > threshold).astype(int)

# Interpret the result for the single image
if prediction >= threshold:
    print(f"🔴 The model predicts this image is **FORGED** with confidence: {prediction:.4f}")
else:
    print(f"🟢 The model predicts this image is **AUTHENTIC** with confidence: {1 - prediction:.4f}")


# %%
import matplotlib.pyplot as plt

# Predict probabilities on the test set
y_pred_proba = model.predict(X_test)

# Plot histogram of predicted probabilities
plt.figure(figsize=(8, 5))
plt.hist(y_pred_proba, bins=20, color='skyblue', edgecolor='black', alpha=0.75)
plt.xlabel("Predicted Probability (Forged Likelihood)")
plt.ylabel("Number of Images")
plt.title("MISD Dataset - Distribution of Predicted Probabilities")
plt.grid(True)
plt.show()


# %%
# Get flattened predicted probabilities
y_pred_proba = model.predict(X_test).flatten()

# Print the first 20 predictions with interpretation
print("MISD Dataset - Sample Predictions on Test Set:")
for i in range(20):
    label = "FORGED" if y_pred_proba[i] >= 0.5 else "AUTHENTIC"
    print(f"Image {i+1}: Probability = {y_pred_proba[i]:.4f}, Classified as: {label}")


# %%
import numpy as np

# MISD Dataset - Count of Authentic vs Forged Images
print("MISD Dataset Summary:")
print("Total Authentic Images:", np.sum(all_labels == 0))
print("Total Forged Images:", np.sum(all_labels == 1))


# %%



