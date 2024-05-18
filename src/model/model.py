from tensorflow import keras 
from keras import models, layers, regularizers
import matplotlib as plt
from data_compiler import compile_data

NUM_CLASSES = 100
NUM_EXAMPLES_PER_CLASS = 4000
TOTAL_EXAMPLES = NUM_CLASSES * NUM_EXAMPLES_PER_CLASS

TRAIN_PROP = 0.8
TRAIN_IDX = int(TOTAL_EXAMPLES * TRAIN_PROP)

X, y, classes = compile_data(NUM_CLASSES, NUM_EXAMPLES_PER_CLASS)
X_train, y_train = X[:TRAIN_IDX], y[:TRAIN_IDX]
X_testing, y_testing = X[TRAIN_IDX:], y[TRAIN_IDX:]
test_split = len(X_testing)//2
X_val, y_val = X_testing[:test_split], y_testing[:test_split]
X_test, y_test = X_testing[test_split:], y_testing[test_split:]


print("starting model")
#model yields 0.7263 train accuracy, 0.7546 val accuracy
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
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# To be used on final model
#loss, accuracy = model.evaluate(X_test, y_test)
#print(loss, accuracy)