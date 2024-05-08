import cv2 as cv
import tensorflow as tf
import matplotlib as plt
import IPython.display as display
import numpy as np
import os

# filename = '../data/quickdraw_tutorial_dataset_v1.tar.gz'
# dataset = tf.data.TFRecordDataset(filename)
# print(type(dataset))
NUM_CLASSES = 345
NUM_EXAMPLES_PER_CLASS = 2
VECTOR_LENGTH = 784

directory = os.getcwd() + "\\Quick-Draw-Guesser\\data\\draw_data"

def compile_data(num_classes, num_examples_per_class):
    '''
    Compiles all individual classes of the Quick Draw data into a single 
    matrix and corresponding target vector to feed into the CNN and shuffles it. 
    Additionally returns a string array of all class names.
    '''
    X = np.zeros((num_classes * num_examples_per_class, VECTOR_LENGTH))
    y = np.zeros(num_classes * num_examples_per_class)
    classes = []

    for class_enum, file in enumerate(os.listdir(directory)):
        filename = os.fsdecode(file)
        path = directory + "\\" + filename
        arr = np.load(path)
        for i in range(num_examples_per_class):
            X[class_enum * num_examples_per_class + i] = arr[i] / 255
            y[class_enum * num_examples_per_class + i] = class_enum
            classes.append(filename[:-4])

    # TODO: shuffle data
    return X, y, classes

compile_data(NUM_CLASSES, NUM_EXAMPLES_PER_CLASS)