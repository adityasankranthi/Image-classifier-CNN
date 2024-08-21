import numpy as np
from keras.preprocessing import image_dataset_from_directory
from keras.applications import VGG16
from keras.layers import Dense, Flatten, GlobalAveragePooling2D
from keras.models import Model
from keras.src.layers import Dropout
from sklearn.metrics import confusion_matrix

train_dir = 'Monkey Species Data/Training Data'
test_dir = 'Monkey Species Data/Prediction Data'

train_set = image_dataset_from_directory(train_dir, label_mode='categorical', image_size=(100, 100))
test_set = image_dataset_from_directory(test_dir, label_mode='categorical', image_size=(100, 100), shuffle=False)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
for layer in base_model.layers:
    layer.trainable = False
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

fine_tuned_model = Model(inputs=base_model.input, outputs=predictions)
fine_tuned_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
epochs = 20
fine_tuned_model.fit(train_set, epochs=epochs, validation_data=test_set)
test_loss, test_acc = fine_tuned_model.evaluate(test_set)
print(f'Test accuracy for fine-tuned model: {test_acc}')
test_predictions_fine_tuned = fine_tuned_model.predict(test_set)
test_labels = np.concatenate([labels for images, labels in test_set], axis=0)
confusion_matrix_fine_tuned = confusion_matrix(test_labels.argmax(axis=1), test_predictions_fine_tuned.argmax(axis=1))
print(f'Confusion matrix for fine-tuned model:\n{confusion_matrix_fine_tuned}')

fine_tuned_model.save('fine_tuned_model.keras')
