import numpy as np
import os
from sklearn.utils import shuffle

IMAGE_DIM = 28

directory = os.getcwd() + '\\Quick-Draw-Guesser\\data\\draw_data'

def compile_data(num_classes, num_train_examples_per_class, num_testval_examples_per_class):
    '''
    Compiles all individual classes of the Quick Draw data into matrices and corresponding 
    target vectors for train, validation and testing to feed into the CNN and shuffles it. 
    Additionally returns a string array of all class names.
    '''
    np.random.seed(1)
    X_train = np.empty((num_classes * num_train_examples_per_class, IMAGE_DIM, IMAGE_DIM))
    y_train = np.zeros(num_classes * num_train_examples_per_class)
    X_val = np.empty((num_classes * num_testval_examples_per_class, IMAGE_DIM, IMAGE_DIM))
    y_val = np.zeros(num_classes * num_testval_examples_per_class)
    X_test = np.empty((num_classes * num_testval_examples_per_class, IMAGE_DIM, IMAGE_DIM))
    y_test = np.zeros(num_classes * num_testval_examples_per_class)
    classes = []

    for class_enum, file in enumerate(os.listdir(directory)):
        if class_enum == num_classes - 1:
            break
        filename = os.fsdecode(file)
        path = directory + '\\' + filename
        arr = np.load(path)

        # training data
        for i in range(num_train_examples_per_class):
            X_train[class_enum * num_train_examples_per_class + i] = np.reshape(arr[i] / 255, (28, 28))
            y_train[class_enum * num_train_examples_per_class + i] = class_enum
            classes.append(filename[:-4])

        # validation data
        offset = num_train_examples_per_class
        for i in range(num_testval_examples_per_class):
            X_val[class_enum * num_testval_examples_per_class + i] = np.reshape(arr[i + offset] / 255, (28, 28))
            y_val[class_enum * num_testval_examples_per_class + i] = class_enum

        # test data
        offset = num_train_examples_per_class + num_testval_examples_per_class
        for i in range(num_testval_examples_per_class):
            X_test[class_enum * num_testval_examples_per_class + i] = np.reshape(arr[i + offset] / 255, (28, 28))
            y_test[class_enum * num_testval_examples_per_class + i] = class_enum
            
    y_train = y_train.astype(int)
    X_train, y_train = shuffle(X_train, y_train)

    y_val = y_val.astype(int)
    X_val, y_val = shuffle(X_val, y_val)

    y_test = y_test.astype(int)
    X_test, y_test = shuffle(X_test, y_test)
    return X_train, y_train, X_val, y_val, X_test, y_test, classes