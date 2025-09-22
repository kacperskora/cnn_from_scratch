import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_accuracy_curve(accuracy_history, val_accuracy_history): 
    plt.plot(accuracy_history, 'b', label = 'Training accuracy')
    plt.plot(val_accuracy_history, 'r', label = 'Validation history')
    plt.xlabel('Iteration', fontsize = 16)
    plt.ylabel('Accuracy rate', fontsize = 16)
    plt.legend()
    plt.title('Training accuracy', fontsize = 16)
    plt.show()


def plot_learning_curve(loss_history): 
    plt.plot(loss_history, 'b', label = 'Cross entropy')
    plt.xlabel('Iteration', fontsize = 16)
    plt.ylabel('Loss', fontsize = 16)
    plt.legend()
    plt.title('Learning Curve', fontsize = 16)
    plt.show()


def plot_histogram(layer_name, layer_weights): 
    plt.hist(layer_weights)
    plt.title('Histogram of ' + str(layer_name))
    plt.xlabel('Value')
    plt.ylabel('Number')
    plt.show()
    
    
def plot_sample(image, true_label, predicted_label): 
    plt.imshow(image)
    if true_label and predicted_label is not None: 
        if type(true_label) == 'int':
            plt.title('True label: %d, Predicted label: %d' % (true_label, predicted_label))
        else: 
            plt.title('True label: %s, Predicted label: %s' % (true_label, predicted_label))
    plt.show()
    

