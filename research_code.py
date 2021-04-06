
from matplotlib import pyplot as plt
# import matplotlib.image as mpimg
import numpy as np
# import imageio as im
from keras import models, regularizers, losses, optimizers
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, MaxPool2D
from keras.layers import Flatten
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Dropout, InputLayer
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from keras.utils.np_utils import to_categorical
# from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.optimizers import Adam
import tensorflow as tf
from keras.models import Model
import datetime
from matplotlib import pyplot
from numpy import expand_dims
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img


def read_data(train, test):

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Train data
    training_set = train_datagen.flow_from_directory(str(train),
                                                     target_size = (224, 224),
                                                     batch_size = 16,
                                                     shuffle=False,
                                                     class_mode = 'categorical')

    # get the class labels for the training data, in the original order
    train_labels = training_set.classes

    # convert the training labels to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=2)

    # Test data
    test_set = test_datagen.flow_from_directory(str(test),
                                                target_size = (224, 224),
                                                batch_size = 16,
                                                shuffle=False,
                                                class_mode = 'categorical')

    # get the class labels for the training data, in the original order
    test_labels = test_set.classes

    # convert the training labels to categorical vectors
    test_labels = to_categorical(test_labels, num_classes=2)

    return training_set, test_set, train_labels, test_labels


def train_model(training_set, test_set, epc = 10, stp = 100):
    # Initialize the model
    classifier = Sequential()

    # Load the VGG model
    # load model without output layer
    vgg_model = VGG16(
        weights='E:/Projects/Dialog/Models/pre_trained_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        include_top=False, input_shape=(224, 224, 3))

    classifier.add(vgg_model)

    classifier.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
    classifier.add(Conv2D(32, (3, 3), activation='relu'))
    classifier.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    classifier.add(Flatten())  # Flatten the input

    classifier.add(Dense(512, activation='relu'))
    classifier.add(Dense(512, activation='relu'))
    classifier.add(Dense(512, activation='relu'))
    classifier.add(Dense(2, activation='softmax'))


    # Model summary
    print(classifier.summary())

    optimizer = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)

    classifier.compile(optimizer= optimizer, #'rmsprop',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    # To store the weights of the best performing epoch
    checkpointer = ModelCheckpoint(filepath="best_weights.hdf5",
                                   monitor = 'val_accuracy',
                                   verbose=1,
                                   save_best_only=True)

    annealer = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-5)
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # Train the model
    history = classifier.fit_generator(training_set,
                                       steps_per_epoch = int(stp),
                                       epochs = int(epc),
                                       callbacks=[annealer, checkpointer],#tensorboard_callback],
                                       validation_data = test_set,
                                       validation_steps = 50)
    return history, classifier


def save_model(classifier, name):

    # Save the model
    classifier.save(str(name))


def read_model(name):

    # Load the model
    model = load_model(str(name))
    return model


def accuracy_curves(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def visualize_activalions(img_path, classifier):

    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.

    print(img_tensor.shape)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = classifier.predict_classes(images, batch_size=10)
    print("Predicted class is: ",classes)


    # Visualizing intermediate activations
    layer_outputs = [layer.output for layer in classifier.layers[:12]] # Extracts the outputs of the top 12 layers
    activation_model = models.Model(inputs=classifier.input, outputs=layer_outputs)
    activations = activation_model.predict(img_tensor)
    first_layer_activation = activations[0]
    print(first_layer_activation.shape)

    # plt.matshow(first_layer_activation[0, :, :, 4], cmap='viridis')

    layer_names = []
    for layer in classifier.layers[:12]:
        layer_names.append(layer.name)

    images_per_row = 16

    for layer_name, layer_activation in zip(layer_names, activations):  # Displays the feature maps
        n_features = layer_activation.shape[-1]  # Number of features in the feature map
        size = layer_activation.shape[1]  # The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row  # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols):  # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                channel_image = layer_activation[0,
                                :, :,
                                col * images_per_row + row]
                channel_image -= channel_image.mean()
                if int(channel_image.std()) != 0:
                    channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size: (col + 1) * size,  # Displays the grid
                row * size: (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()


def visualize_feature_map(img_path, model):
    # redefine model to output right after the first hidden layer
    ixs = [2,5,7,9]
    outputs = [model.layers[i].output for i in ixs]
    model = Model(inputs=model.inputs, outputs=outputs)
    # load the image with the required shape
    img = load_img(img_path, target_size=(224, 224))
    # convert the image to an array
    img = img_to_array(img)
    # expand dimensions so that it represents a single 'sample'
    img = expand_dims(img, axis=0)
    # prepare the image (e.g. scale pixel values for the vgg)
    img = preprocess_input(img)
    # get feature map for first hidden layer
    feature_maps = model.predict(img)
    # plot the output from each block
    square = 5
    for fmap in feature_maps:
        # plot all 64 maps in an 8x8 squares
        ix = 1
        for _ in range(square):
            for _ in range(square):
                # specify subplot and turn of axis
                ax = pyplot.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                pyplot.imshow(fmap[0, :, :, ix - 1])
                ix += 1
        # show the figure
        pyplot.show()


def train_validation(history):

    #Graphing our training and validation

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()


def get_confusion_matrix(classifier, test_set, test_labels):
    # Confution Matrix and Classification Report
    actual_classes = test_set.classes
    actual_class_labels = list(test_set.class_indices.keys())
    # predictions = classifier.predict_generator(test_set, steps=len(test_set))
    predictions = classifier.predict(x=test_set, steps=len(test_set), verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    print('Confusion Matrix')
    conf = confusion_matrix(actual_classes, predicted_classes)
    print(conf)
    print('Classification Report')
    report = classification_report(actual_classes, predicted_classes, target_names=actual_class_labels)
    print(report)


# Main - Build model


train = 'E:/MSc/Research/Data/test case 2/train/'
test = 'E:/MSc/Research/Data/test case 2/test/'


# training_set, test_set, train_labels, test_labels = read_data(train, test)
# history, classifier = train_model(training_set, test_set, epc = 1, stp = 1)
# accuracy_curves(history)
# train_validation(history)
# save_model(classifier, 'E:/MSc/Research/Models/test case 54312.h5')

# classifier = read_model('E:/MSc/Research/Models/old models/test_case_2.h5')
# get_confusion_matrix(classifier, test_set, test_labels)
# get_confusion_matrix(classifier, training_set, train_labels)

# classifier = VGG16(
#     weights='E:/Projects/Dialog/Models/pre_trained_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
#     include_top=False, input_shape=(224, 224, 3))
#
classifier = VGG16(
        weights='E:/Projects/Dialog/Models/pre_trained_models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        include_top=False, input_shape=(224, 224, 3))
img_path = "E:/MSc/Research/Data/test case 1/test/invalid/dtv_88888888_1571310474440_1_1571310629067.jpg"
visualize_feature_map(img_path, classifier)
