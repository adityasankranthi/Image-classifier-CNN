import glob
import os
import numpy as np
from keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.metrics import confusion_matrix

# Removing non-JFIF files
files = glob.glob("Monkey Species Data/*/*/*")
for file in files:
    with open(file, "rb") as f:
        if not b"JFIF" in f.peek(10):
            os.remove(file)

train_dir = 'Monkey Species Data/Training Data'
test_dir = 'Monkey Species Data/Prediction Data'

train_set = image_dataset_from_directory(train_dir, label_mode='categorical', image_size=(100, 100))
test_set = image_dataset_from_directory(test_dir, label_mode='categorical', image_size=(100, 100), shuffle=False)

# CNN Architecture 1
model1 = Sequential()
model1.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model1.add(MaxPooling2D((2, 2)))
model1.add(Conv2D(64, (3, 3), activation='relu'))
model1.add(MaxPooling2D((2, 2)))
model1.add(Flatten())
model1.add(Dense(128, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(10, activation='softmax'))

model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 20
model1.fit(train_set, epochs=epochs, validation_data=test_set)
test_loss1, test_acc1 = model1.evaluate(test_set)
print(f'Test accuracy for model 1: {test_acc1}')
test_predictions_model1 = model1.predict(test_set)

# CNN Architecture 2
model2 = Sequential()
model2.add(Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model2.add(MaxPooling2D((4, 4)))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(256, (3, 3), activation='relu'))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(10, activation='softmax'))

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 25
model2.fit(train_set, epochs=epochs, validation_data=test_set)
test_loss2, test_acc2 = model2.evaluate(test_set)
print(f'Test accuracy for model 2: {test_acc2}')
test_predictions_model2 = model2.predict(test_set)

test_labels = np.concatenate([labels for images, labels in test_set], axis=0)
confusion_matrix_model1 = confusion_matrix(test_labels.argmax(axis=1), test_predictions_model1.argmax(axis=1))
print(f'Confusion matrix for model 1:\n{confusion_matrix_model1}')

confusion_matrix_model2 = confusion_matrix(test_labels.argmax(axis=1), test_predictions_model2.argmax(axis=1))
print(f'Confusion matrix for model 2:\n{confusion_matrix_model2}')

if test_acc1 > test_acc2:
    better_model = model1
else:
    better_model = model2
better_model.save('better_model.keras')
