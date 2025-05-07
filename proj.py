import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------ Dataset Paths ------------------
train_dir = 'D:\\new\\train'        # Replace with actual path
val_dir = 'D:\\new\\test'     # Replace with actual path

# ------------------ Data Preprocessing & Augmentation ------------------
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,       # Increased rotation
    zoom_range=0.3,          # More zoom variation
    shear_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,      # Flip images vertically
    brightness_range=[0.8, 1.2] # Adjust brightness levels
)
 
val_datagen = ImageDataGenerator(rescale=1.0 / 255)


train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)


val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='categorical'
)

print("Class indices:", train_generator.class_indices)

# ------------------ Load Pretrained MobileNetV2 ------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False  # Freeze initial layers

# ------------------ Custom Classification Head ------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduce dimensions
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Dropout to prevent overfitting
output_layer = Dense(len(train_generator.class_indices), activation='softmax')(x)

# Final Model
model = Model(inputs=base_model.input, outputs=output_layer)

# ------------------ Compile Model ------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # Lower learning rate for stability
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ------------------ Compute Class Weights (For Imbalanced Data) ------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# ------------------ Train Model with Early Stopping ------------------
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy', patience=8, restore_best_weights=True
)

history = model.fit(  
    train_generator,
    validation_data=val_generator, 
    epochs=40,  # Start with 30 epochs 
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    class_weight=class_weights,
    callbacks=[early_stopping]
)

# ------------------ Fine-Tuning: Unfreeze Some Layers & Retrain ------------------
base_model.trainable = True  # Unfreeze base model for fine-tuning

model.compile(
    optimizer=Adam(learning_rate=1e-5),  # Even lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=25,  # Fine-tune for 10 more epochs
    steps_per_epoch=len(train_generator),
    validation_steps=len(val_generator),
    callbacks=[early_stopping]
)

# ------------------ Save Model ------------------
model.save('eye_disease_model.h5')

# ------------------ Evaluate Model ------------------
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Final Validation Accuracy: {val_accuracy * 100:.2f}%")

# ------------------ Classification Report ------------------
val_preds = model.predict(val_generator)
val_preds_classes = np.argmax(val_preds, axis=1)
true_classes = val_generator.classes
class_labels = list(train_generator.class_indices.keys())

print(classification_report(true_classes, val_preds_classes, target_names=class_labels))

# ------------------ Plot Accuracy & Loss ------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history_finetune.history['accuracy'], label='Fine-Tuned Train Accuracy', linestyle='dashed')
plt.plot(history_finetune.history['val_accuracy'], label='Fine-Tuned Validation Accuracy', linestyle='dashed')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.plot(history_finetune.history['loss'], label='Fine-Tuned Train  Loss', linestyle='dashed')
plt.plot(history_finetune.history['val_loss'], label='Fine-Tuned Validation Loss', linestyle='dashed')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ------------------ Prediction on New Image ------------------
from tensorflow.keras.preprocessing import image

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    return predicted_class, confidence   

# Example: Test on a New Image
img_path = 'D:\\new\\eye diseases\\diabetic_retinopathy\\1052_left.jpeg'  # Replace with test image path #cataract image i give
predicted_class, confidence = predict_image(img_path)
print(f"Predicted Class: {predicted_class}, Confidence: {confidence:.2f}")
