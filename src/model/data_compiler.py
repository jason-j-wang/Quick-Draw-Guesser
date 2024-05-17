import numpy as np
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

IMAGE_DIM = 28

directory = os.getcwd() + "\\Quick-Draw-Guesser\\data\\draw_data"

def compile_data(num_classes, num_examples_per_class):
    '''
    Compiles all individual classes of the Quick Draw data into a single 
    matrix and corresponding target vector to feed into the CNN and shuffles it. 
    Additionally returns a string array of all class names.
    '''
    np.random.seed(1)
    X = np.empty((num_classes * num_examples_per_class, IMAGE_DIM, IMAGE_DIM))
    y = np.zeros(num_classes * num_examples_per_class)
    classes = []

    for class_enum, file in enumerate(os.listdir(directory)):
        if class_enum == num_classes - 1:
            break
        filename = os.fsdecode(file)
        path = directory + "\\" + filename
        arr = np.load(path)
        #print(filename)
        for i in range(num_examples_per_class):
            X[class_enum * num_examples_per_class + i] = np.reshape(arr[i] / 255, (28, 28))
            y[class_enum * num_examples_per_class + i] = class_enum
            classes.append(filename[:-4])

    y = y.astype(int)
    X, y = shuffle(X, y)
    return X, y, classes

# X, y, classes = compile_data(345, 1)
# ex = X[0]
# label = y[0]
# print(classes[label])

# plt.imshow(ex, cmap=plt.cm.binary)
# plt.show()