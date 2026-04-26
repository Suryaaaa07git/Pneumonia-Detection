# ================================
# 1. IMPORT LIBRARIES
# ================================
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# ================================
# 2. DATASET PATH
# ================================
train_dir = "dataset/train"
test_dir = "dataset/test"

IMG_SIZE = 224
BATCH_SIZE = 32

# ================================
# 3. DATA PREPROCESSING
# ================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# ================================
# 4. BUILD CNN MODEL
# ================================
model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ================================
# 5. TRAIN MODEL
# ================================
history = model.fit(
    train_data,
    epochs=10,
    validation_data=test_data
)

# ================================
# 6. EVALUATE MODEL
# ================================
loss, accuracy = model.evaluate(test_data)
print(f"\nTest Accuracy: {accuracy*100:.2f}%")

# ================================
# 7. CONFUSION MATRIX
# ================================
Y_pred = model.predict(test_data)
y_pred = np.argmax(Y_pred, axis=1)

y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:\n", cm)

# Plot confusion matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal','Pneumonia'],
            yticklabels=['Normal','Pneumonia'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

os.makedirs("images", exist_ok=True)
plt.savefig("images/confusion_matrix.png")
plt.show()

# ================================
# 8. CLASSIFICATION REPORT
# ================================
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=['Normal','Pneumonia']))

# ================================
# 9. SAMPLE X-RAY PREDICTION
# ================================
def predict_image(img_path):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224,224))
    img_normalized = img_resized / 255.0
    img_input = np.reshape(img_normalized, (1,224,224,3))

    prediction = model.predict(img_input)

    label = np.argmax(prediction)

    if label == 0:
        result = "Normal"
    else:
        result = "Pneumonia"

    # Show image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Prediction: {result}")
    plt.axis('off')

    plt.savefig("images/sample_prediction.png")
    plt.show()

    return result

# ================================
# 10. TEST WITH IMAGE
# ================================
test_image_path = "test_image.jpg"  # change this path
print("\nSample Prediction:", predict_image(test_image_path))