import os
from tensorflow import keras 
from keras import models, layers, regularizers
from data_compiler import compile_data

NUM_CLASSES = 100
NUM_TRAIN_EXAMPLES_PER_CLASS = 4000
NUM_TESTVAL_EXAMPLES_PER_CLASS = 500

X_train, y_train, X_val, y_val, X_test, y_test, classes = compile_data(NUM_CLASSES, NUM_TRAIN_EXAMPLES_PER_CLASS, NUM_TESTVAL_EXAMPLES_PER_CLASS)

#model yields 0.7397 train accuracy, 0.7678 val accuracy, 0.8107 test accuracy
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(256, (3, 3), activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu', kernel_regularizer=regularizers.L1L2()))
model.add(layers.Dropout(0.2))

model.add(layers.Dense(100, activation='softmax'))

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val))

model_file = 'model_v1.keras'
model_directory = os.getcwd() + '\\Quick-Draw-Guesser\\src\\model\\checkpoint\\' + model_file
model.save(model_directory)

# To be used on final model
loss, accuracy = model.evaluate(X_test, y_test)
print(loss, accuracy)