# Import necessary libraries
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

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

# Compile the model
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 20
model1.fit(training_set, epochs=epochs, validation_data=test_set)

# Evaluate the model on test data
test_loss, test_acc = model1.evaluate(test_set)
print(f'Test accuracy for model 1: {test_acc}')

# Confusion matrix for model 1
predictions = model1.predict(test_set)
confusion_matrix = sklearn.metrics.confusion_matrix(test_set.classes, predictions.argmax(axis=1))
print(f'Confusion matrix for model 1:\n{confusion_matrix}')

# Save the model
model1.save('model1.h5')

# CNN Architecture 2
model2 = Sequential()
model2.add(Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(32, (3, 3), activation='relu'))
model2.add(MaxPooling2D((2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(10, activation='softmax'))

# Compile the model
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 20
model2.fit(training_set, epochs=epochs, validation_data=test_set)

# Evaluate the model on test data
test_loss, test_acc = model2.evaluate(test_set)
print(f'Test accuracy for model 2: {test_acc}')

# Confusion matrix for model 2
predictions = model2.predict(test_set)
confusion_matrix = sklearn.metrics.confusion_matrix(test_set.classes, predictions.argmax(axis=1))
print(f'Confusion matrix for model 2:\n{confusion_matrix}')

# Save the better model
if test_acc_model1 > test_acc_model2:
    better_model = model1
else:
    better_model = model2
better_model.save('better_model.h5')