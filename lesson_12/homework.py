#!/usr/bin/env python3

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input

plt.rcParams['figure.figsize'] = [15, 6]

# constants
data_folder_path = "data/subset_homework"
class_0_folder_name = "class_id_0"
class_1_folder_name = "class_id_1"
number_of_random_images_to_print = 20
number_of_models_to_create = 5

# load the subset data
def load_data_subset(data_folder_path, folder_to_load, class_idx):
    fnames = os.listdir(os.path.join(data_folder_path, folder_to_load))
    images = [cv2.imread(os.path.join(data_folder_path, folder_to_load, image_name), cv2.IMREAD_UNCHANGED) for image_name in fnames]
    labels = [class_idx] * len(images)

    return fnames, images, labels

def plot_random_images_from_set(image_set, labels, nb_pictures_to_pick):
    plt.figure()

    if (nb_pictures_to_pick < 0 or nb_pictures_to_pick >= len(image_set)):
        print("[plot_random_images_from_set] invlid number provided")
        return

    for cnt, idx in enumerate(np.random.randint(0, len(image_set), nb_pictures_to_pick)):
        plt.subplot(2, np.int32(nb_pictures_to_pick / 2), cnt + 1)
        plt.imshow(image_set[idx], cmap='gray', vmin=0, vmax=255)
        plt.title(labels[idx])
        plt.axis(False)

    return

def shuffle_data(images_1, labels_1, images_2, labels_2):
    images = images_1 + images_2
    labels = labels_1 + labels_2
    images, labels = shuffle(images, labels)

    images = np.array(images)
    labels = np.array(labels)

    print("Images", images.shape)
    print("Labels", labels.shape)

    return images, labels

def flatten_data(images):
    flatten_images = np.array([image.flatten() for image in images]) / 255
    print("Pixels: ", flatten_images.shape)
    return flatten_images

# function that builds the nn with n number of hidden layers
# in our case by calculating the number of neurons for the next layer we do expect to have a valid number
# for the testing purposes
def build_nn(pixels, number_of_layers):
    inputs = Input(shape=(pixels.shape[1],))
    # create a simple nn, if number is bigger we add hidden layers
    if number_of_layers == 0:
        outputs = Dense(1, activation="linear")(inputs)
        model = Model(inputs, outputs)
        return model

    # lets calculate number of neurons
    neurons_number = int(pixels.shape[1] / 2)
    # create layers
    hidden_layer = Dense(neurons_number, activation="relu")(inputs)
    # now lets create remaining hidden layers of nn
    for i in range(1, number_of_layers):
         # lets update number of neurons, each subsequent layer will recieve half the neurons number from previous layer
        neurons_number =  int(neurons_number / 2)
        hidden_layer = Dense(neurons_number, activation="relu")(hidden_layer)
    outputs = Dense(1, activation="linear")(hidden_layer)
    model = Model(inputs, outputs)

    return model

# function that creates n number of models and trains k number of epochs
def create_models(pixels, labels, epochs):
    models = []
    for i in range(0, number_of_models_to_create):
        models.append(build_nn(pixels, i))
        # print summary
        models[i].summary()
        # plot model
        tf.keras.utils.plot_model(models[i], to_file="model_" + str(i) + ".png", show_shapes=True, show_layer_names=True)
        # lets use defalut optimizer and loss function as in a lecture
        models[i].compile(optimizer="adam", loss="mean_squared_error")
        models[i].fit(pixels, labels, epochs=epochs, batch_size=32)

    return models

def main(args):
    # load class_0 images and labels
    fnames_0, images_0, labels_0 = load_data_subset(data_folder_path, class_0_folder_name, 0)
    # load class_1 images and labels
    fnames_1, images_1, labels_1 = load_data_subset(data_folder_path, class_1_folder_name, 1)

    print('Num samples class_0', len(images_0))
    print('Num samples class_1', len(images_1))

    # plot random igames from both sets
    plot_random_images_from_set(images_0, labels_0, number_of_random_images_to_print)
    plot_random_images_from_set(images_1, labels_1, number_of_random_images_to_print)
    plt.show()

    # shuffle and flatten the data
    images, labels = shuffle_data(images_0, labels_0, images_1, labels_1)
    pixels = flatten_data(images)
    print(pixels.shape)

    plot_random_images_from_set(images, labels, number_of_random_images_to_print)
    plt.show()

    # lets make a list of epochs to train
    # we will create 5 models so we will test on number of epochs = (i + 1) * 10
    # meaning 10, 20 ....
    epochs_to_train = []
    for i in range (number_of_models_to_create):
        epochs_to_train.append((i + 1) *10)

    # create models and train them different amount of epochs
    models_to_epochs = []
    for i in range (number_of_models_to_create):
        models_to_epochs.append(create_models(pixels, labels, epochs_to_train[i]))

    print("***************************************************************")
    # now lets go through all our trained models and check accuracy of each of them and find the more accurate for the epoch
    i = 0
    most_accuracy = 0
    most_accurate_model = None
    epochs_model_trained = 0
    most_accurate_model_predictions = []
    for models in models_to_epochs:
        accuracies = []
        models_predictions = []
        for model in models:
            predictions = model.predict(pixels).squeeze()
            predictions = predictions > 0.5
            correct = 0
            models_predictions.append(predictions)
            for prediction, label in zip(predictions, labels):
                if prediction == label:
                    correct = correct + 1
            accuracies.append(correct/len(labels))
        max_accuracy_index = np.argmax(accuracies)
        if accuracies[max_accuracy_index] > most_accuracy:
            most_accuracy = accuracies[max_accuracy_index]
            most_accurate_model = model
            epochs_model_trained = (i + 1) * 10
            most_accurate_model_predictions = models_predictions[i]
        print("ACCURACIES FOR " + str((i + 1) * 10) + " EPOCHS: ", accuracies, " MAX ACCURACY IS MODEL WITH ", max_accuracy_index, " LAYERS, VALUE: ", accuracies[max_accuracy_index])
        i = i + 1
    print("***************************************************************")

    print("MOST ACCURATE MODEL TRAINED FOR ", epochs_model_trained, " AND HAS ACCURACY: ", most_accuracy)
    most_accurate_model.summary()
    # test the results on the most trained model
    for cnt, idx in enumerate(np.random.randint(0, len(images), 10)): 
        plt.subplot(2,5,cnt+1), plt.imshow(images[idx], cmap='gray', vmin=0, vmax=255)
        plt.title('Label: ' + str(labels[idx]) + ' | Prediction: ' + str(most_accurate_model_predictions[idx]))
        plt.axis(False)
    plt.show()

    return 
 
if __name__ == '__main__':
    main(sys.argv)