import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.src.utils import image_dataset_from_directory

better_model = load_model('better_model.keras')
fine_tuned_model = load_model('fine_tuned_model.keras')
test_dir = 'Monkey Species Data/Prediction Data'
test_set = image_dataset_from_directory(test_dir, label_mode='categorical', image_size=(100, 100), shuffle=False)
test_predictions_model1 = better_model.predict(test_set)

test_images = []
test_labels = []
for images, labels in test_set:
    test_images.extend(images)
    test_labels.extend(labels)

test_images = np.array(test_images)
test_labels = np.array(test_labels)
incorrect_indices = np.where(test_labels.argmax(axis=1) != test_predictions_model1.argmax(axis=1))[0]
random_indices = np.random.choice(incorrect_indices, size=10, replace=False)
class_names = sorted(os.listdir('Monkey Species Data/Training Data'))

for index in random_indices:
    fig = plt.figure(figsize=(10, 6))
    image = test_images[index]
    correct_label = class_names[np.argmax(test_labels[index])]
    predicted_label_model1 = class_names[np.argmax(test_predictions_model1[index])]
    predicted_label_fine_tuned = class_names[np.argmax(fine_tuned_model.predict(np.expand_dims(image, axis=0)))]
    image_display = np.uint8(image)
    plt.imshow(image_display)
    plt.title(f'Correct: {correct_label}, Model 1 Prediction: {predicted_label_model1}, Fine-tuned Model Prediction: {predicted_label_fine_tuned}')
    plt.axis('off')
    plt.show()
