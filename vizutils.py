import pandas as pd
from matplotlib import pyplot as plt

def plot_performance_Loss(history_path):
    # Load the training history from the CSV file
    history = pd.read_csv(history_path)

    # Plot the performance
    fig = plt.figure()
    plt.plot(history['loss'], color='teal', label='loss')
    plt.plot(history['val_loss'], color='orange', label='val_loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()


def plot_performance_accuracy(history_path):
    # Load the training history from the CSV file
    history = pd.read_csv(history_path)

    # Plot the performance
    fig = plt.figure()
    plt.plot(history['categorical_accuracy'], color='teal', label='accuracy')
    plt.plot(history['val_categorical_accuracy'], color='orange', label='val_accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()