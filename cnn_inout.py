import os 
import idx2numpy
import cv2
import platform
import numpy as np

def load_mnist(): 
    x_train = idx2numpy.convert_from_file('D:/python/nn/train-images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('D:/python/nn/train-labels.idx1-ubyte')
    x_test = idx2numpy.convert_from_file('D:/python/nn/t10k-images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('D:/python/nn/t10k-labels.idx1-ubyte')
    
    train_images =[]                                                # reshaping train images so they are 
    for i in range(x_train.shape[0]):                               # (60000, 1, 28, 28) shape
        train_images.append(np.expand_dims(x_train[i], axis = 0))
    train_images = np.array(train_images)
    
    test_images =[]                                                 # reshaping test images so they are 
    for i in range(x_test.shape[0]):                                # (10000, 1, 28, 28) shape
        test_images.append(np.expand_dims(x_test[i], axis = 0))
    test_images = np.array(test_images)
    
    # shuffling the dataset
    indices = np.random.permutation(train_images.shape[0])
    training_idx, validation_idx = indices[: 55000], indices[55000: ]
    train_images, validation_images = train_images[training_idx, :], train_images[validation_idx, :]
    train_labels, validation_labels = train_labels[training_idx], train_labels[validation_idx]
    
    return {
        'train_images' : train_images, 
        'train_labels' : train_labels, 
        'validation_images' : validation_images, 
        'validation_labels' : validation_labels, 
        'test_images' : test_images, 
        'test_labels' : test_labels
    }
    

def to_gray(image_name):
        image = cv2.imread(image_name + '.png')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow('Gray image', image)
        cv2.imwrite(image_name + '.png', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    
def minmax_normalize(x): 
    minx = np.min(x)
    maxx = np.max(x)
    x = ((x-minx) / (maxx - minx))
    return x

def preprocess(dataset):
    dataset['train_images'] = np.array([minmax_normalize(x) for x in dataset['train_images']])
    dataset['validation_images'] = np.array([minmax_normalize(x) for x in dataset['validation_images']])
    dataset['test_images'] = np.array([minmax_normalize(x) for x in dataset['test_images']])
    return dataset
    
         








