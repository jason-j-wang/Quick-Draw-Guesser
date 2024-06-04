import os
from tensorflow import keras 
from keras import models, layers, regularizers
import matplotlib as plt
from data_compiler import compile_data
NUM_CLASSES = 100
NUM_TRAIN_EXAMPLES_PER_CLASS = 1
NUM_TESTVAL_EXAMPLES_PER_CLASS = 500
X_train, y_train, X_val, y_val, X_test, y_test, classes = compile_data(NUM_CLASSES, NUM_TRAIN_EXAMPLES_PER_CLASS, NUM_TESTVAL_EXAMPLES_PER_CLASS)

model_file = 'model_76.keras'
model_directory = os.getcwd() + '\\Quick-Draw-Guesser\\src\\model\\checkpoint\\' + model_file
model = keras.models.load_model(model_directory)
loss, accuracy = model.evaluate(X_test, y_test)
print(loss, accuracy)