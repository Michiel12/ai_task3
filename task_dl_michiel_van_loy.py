import streamlit as st

st.title('AI task 3 by Michiel Van Loy')
st.header("Fruit Recognizer App")

import os
import glob
import matplotlib.pyplot as plt
from PIL import Image

def display_images_and_bar_chart():
  # Specify file locations
  base_directory = "google_images/training_set"
  subdirectories = [d for d in os.listdir(base_directory) if os.path.isdir(os.path.join(base_directory, d))]

  # Dictionary to store image counts for each subdirectory
  image_counts = {}

  # Loop through each subdirectory and count images
  for subdirectory in subdirectories:
      # Use glob to get a list of all image files (adjust the file extension if needed)
      image_files = glob.glob(os.path.join(base_directory, subdirectory, '*.png'))

      # Count the number of images
      image_count = len(image_files)

      # Store the count in the dictionary
      image_counts[subdirectory] = image_count
      
      # Create columns for the images
      col1, col2, col3 = st.columns(3)
      
      # Show the first images of each category
      images_to_show = 3
      for i in range(min(images_to_show, image_count)):
          image_path = image_files[i]
          img = Image.open(image_path)
          # Display the image in the respective column
          if i == 0:
              with col1:
                  st.subheader(f"{subdirectory} {i+1}")
                  st.image(img, width=200)
          elif i == 1:
              with col2:
                  st.subheader(f"{subdirectory} {i+1}")
                  st.image(img, width=200)
          elif i == 2:
              with col3:
                  st.subheader(f"{subdirectory} {i+1}")
                  st.image(img, width=200)

  # Create a bar chart
  plt.figure(figsize=(10, 6))
  plt.bar(image_counts.keys(), image_counts.values())
  plt.xlabel('Categorieën')
  plt.ylabel('Aantal afbeeldingen')
  plt.title('Aantal afbeeldingen per categorie')
  plt.xticks(rotation=45, ha='right')
  # Display the bar chart in Streamlit
  st.pyplot(plt)

st.button("label", on_click=display_images_and_bar_chart)





from keras.preprocessing.image import ImageDataGenerator

# Example usage:
training_set, validation_set, test_set = generate_augmented_data()

def generate_augmented_data():
  # Define directories
  train_dir = 'google_images/training_set'
  test_dir = 'google_images/test_set'

  # Augmentation variables for training set
  train_val_datagen = ImageDataGenerator(validation_split=0.2,
                                    rescale = 1./255,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    vertical_flip= True)

  # Augmentation variables for training set
  test_datagen = ImageDataGenerator(rescale = 1./255)

  # Generate batches of augmented data for training set
  training_set = train_val_datagen.flow_from_directory(train_dir,
                                                  subset='training',
                                                  target_size = (128, 128),
                                                  batch_size = 32,
                                                  class_mode = 'categorical') 

  # Generate batches of augmented data for the validation set
  validation_set = train_val_datagen.flow_from_directory(train_dir,
                                                  subset='validation',
                                                  target_size = (128, 128),
                                                  batch_size = 32,
                                                  class_mode = 'categorical')

  # Generate batches of augmented data for the test set
  test_set = test_datagen.flow_from_directory(test_dir,
                                              target_size = (128, 128),
                                              batch_size = 32,
                                              class_mode = 'categorical')
  
  return training_set, validation_set, test_set





# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras import layers

# Example usage:
model = create_model()

def create_model():
  NUM_CLASSES = 5

  # Create a sequential model with a list of layers
  model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), input_shape = (128, 128, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax")
  ])

  # Compile model
  model.compile(optimizer = optimizers.Adam(learning_rate=0.001), 
                loss = 'categorical_crossentropy', 
                metrics = ['accuracy'])

  print(model.summary())
  
  return model






# Example usage:
trained_history = train_model(model, training_set, validation_set, epochs=25, steps_per_epoch=15)

def train_model(model, training_set, validation_set, epochs=25, steps_per_epoch=15):
  history = model.fit(training_set,
                  validation_data = validation_set,
                  steps_per_epoch = steps_per_epoch,
                  epochs = epochs
                  )
  return history





# Example usage:
test_loss, test_acc = test_accuracy(model, test_set)

def evaluate_test_accuracy(model, test_set):
  # Check test accuracy
  test_loss, test_acc = model.evaluate(test_set)
  print('Test accuracy:', test_acc)
  
  return test_loss, test_acc






import matplotlib.pyplot as plt

# Example usage:
loss_and_accuracy(history)

def loss_and_accuracy(history):
  # Create a figure and a grid of subplots with a single call
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

  # Plot the loss curves on the first subplot
  ax1.plot(history.history['loss'], label='training loss')
  ax1.plot(history.history['val_loss'], label='validation loss')
  ax1.set_title('Loss curves')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss')
  ax1.legend()

  # Plot the accuracy curves on the second subplot
  ax2.plot(history.history['accuracy'], label='training accuracy')
  ax2.plot(history.history['val_accuracy'], label='validation accuracy')
  ax2.set_title('Accuracy curves')
  ax2.set_xlabel('Epoch')
  ax2.set_ylabel('Accuracy')
  ax2.legend()

  # Adjust the spacing between subplots
  fig.tight_layout()

  # Show the figure
  plt.show()







from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Example usage:
plot_confusion_matrix(model, test_set)

def confusion_matrix(model, test_set):
  # Generate predictions for all the test images
  predictions = model.predict(test_set)

  # Extract labels from image file names
  labels = [filename.split('_')[0] for filename in test_set.filenames]

  # Initialize the label encoder
  label_encoder = LabelEncoder()

  # Fit and transform the labels to numerical values
  numeric_labels = label_encoder.fit_transform(labels)

  # First, let's transform all the prediction into the winners (otherwise each prediction gives us the 10 probabilities, but we only need the winner, the one our network thinks it is)
  pred = np.argmax(predictions, axis=1)
  # Now, compare the true labels of the test set, to our predicted winners
  cm = confusion_matrix(numeric_labels, pred)

  # Make the confusion matrix a little more visually attractive
  disp = ConfusionMatrixDisplay(confusion_matrix=cm)

  disp.plot(cmap=plt.cm.Blues)
  plt.show()
