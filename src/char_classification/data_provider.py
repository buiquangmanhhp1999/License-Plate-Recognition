import tensorflow.keras as keras
import numpy as np

from src import data_utils


class Datasets(object):
    def __init__(self):
        self.all_data = []

        # Input data
        self.digits_data = data_utils.get_digits_data('./data/digits.npy')
        self.alphas_data = data_utils.get_alphas_data('./data/alphas.npy')

        # Preprocess
        self.convert_data_format()

    def gen(self):
        np.random.shuffle(self.all_data)
        images = []
        labels = []

        for i in range(len(self.all_data)):
            image, label = self.all_data[i]
            images.append(image)
            labels.append(label)

        labels = keras.utils.to_categorical(labels, num_classes=32)
        return images, labels

    def convert_data_format(self):
        # Digits data
        for i in range(len(self.digits_data)):
            image = self.digits_data[i][0]
            label = self.digits_data[i][1]
            self.all_data.append((image, label))

        # Alpha data
        nb_alphas_data = len(self.alphas_data)
        for i in range(nb_alphas_data * 8):
            image = self.alphas_data[i % nb_alphas_data][0]
            label = self.alphas_data[i % nb_alphas_data][1]
            self.all_data.append((image, label))
