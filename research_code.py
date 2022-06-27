
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from time import time
import wandb
from wandb.keras import WandbCallback
import time

wandb.init(project="image classification", entity="vanodya")


def read_data(train, test, validation):

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    validation_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

    # Train data
    training_set = train_datagen.flow_from_directory(str(train),
                                                     target_size = (224, 224),
                                                     batch_size = 10,
                                                     shuffle=False,
                                                     class_mode = 'categorical')

    # get the class labels for the training data, in the original order
    train_labels = training_set.classes

    # convert the training labels to categorical vectors
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=2)

    # Test data
    test_set = test_datagen.flow_from_directory(str(test),
                                                target_size = (224, 224),
                                                batch_size = 10,
                                                shuffle=False,
                                                class_mode = 'categorical')

    # get the class labels for the training data, in the original order
    test_labels = test_set.classes

    # convert the training labels to categorical vectors
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=2)

    # Validation data
    validation_set = validation_datagen.flow_from_directory(str(validation),
                                                     target_size=(224, 224),
                                                     batch_size=10,
                                                     shuffle=False,
                                                     class_mode='categorical')

    # get the class labels for the training data, in the original order
    validation_labels = validation_set.classes

    # convert the training labels to categorical vectors
    validation_labels = tf.keras.utils.to_categorical(validation_labels, num_classes=2)

    return training_set, test_set, validation_set, train_labels, test_labels, validation_labels


def train_model(training_set, validation_set, epc = 10, stp = 100):
    # Initialize the model
    classifier = tf.keras.models.Sequential()

    # Load the VGG model
    # load model without output layer
    vgg_model = tf.keras.applications.VGG16(
        weights='E:/MSc/Research/Results/VGG/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        include_top=False, input_shape=(224, 224, 3))

    classifier.add(vgg_model)

    classifier.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape=(224, 224, 3)))
    classifier.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    classifier.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    classifier.add(tf.keras.layers.Flatten())  # Flatten the input

    classifier.add(tf.keras.layers.Dense(512, activation='relu'))
    classifier.add(tf.keras.layers.Dense(512, activation='relu'))
    classifier.add(tf.keras.layers.Dense(512, activation='relu'))
    # classifier.add(tf.keras.layers.Dense(512, activation='relu'))
    # classifier.add(tf.keras.layers.Dense(512, activation='relu'))
    classifier.add(tf.keras.layers.Dense(2, activation='softmax'))


    # Model summary
    print(classifier.summary())

    optimizer = tf.keras.optimizers.Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=0.1, decay=0.0)

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="logs\{}".format(time()))

    classifier.compile(optimizer= optimizer, #'rmsprop',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

    # To store the weights of the best performing epoch
    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="best_weights.hdf5",
                                   monitor = 'val_accuracy',
                                   verbose=1,
                                   save_best_only=True)

    annealer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, verbose=1, min_lr=1e-5)


    # Train the model
    history = classifier.fit_generator(training_set,
                                       steps_per_epoch = int(stp),
                                       epochs = int(epc),
                                       callbacks=[annealer, checkpointer, tensorboard, WandbCallback()],
                                       validation_data = validation_set,
                                       validation_steps = len(validation_set))
    return history, classifier


def save_model(classifier, name):

    # Save the model
    classifier.save(str(name))


def read_model(name):

    # Load the model
    model = tf.keras.models.load_model(str(name),compile=False)
    return model


def get_confusion_matrix(classifier, test_set, test_labels):
    # Confution Matrix and Classification Report
    actual_classes = test_set.classes
    actual_class_labels = list(test_set.class_indices.keys())
    predictions = classifier.predict(x=test_set, steps=len(test_set), verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    print('Confusion Matrix')
    conf = confusion_matrix(actual_classes, predicted_classes)
    print(conf)
    print('Classification Report')
    report = classification_report(actual_classes, predicted_classes, target_names=actual_class_labels)
    print(report)


# Main - Read input data

train = 'E:/MSc/Research/satellite verification one shot/data/test case 3/train/'
test = 'E:/MSc/Research/satellite verification one shot/data/test case 3/test/'
validation = 'E:/MSc/Research/satellite verification one shot/data/test case 3/validation/'


# Train and save model

training_set, test_set, validation_set, train_labels, test_labels, validation_labels = read_data(train, test, validation)
history, classifier = train_model(training_set, validation_set, epc = 30, stp = 30)
# save_model(classifier, 'E:/MSc/Research/satellite verification one shot/results siamese/test case 3.h5')


# Load model and get confusion matrix

classifier = read_model('E:/MSc/Research/satellite verification one shot/results siamese/test case 3 CNN vgg16_model_1644731777.1928465.h5')

get_confusion_matrix(classifier, training_set, train_labels)
get_confusion_matrix(classifier, validation_set, validation_labels)
get_confusion_matrix(classifier, test_set, test_labels)


# Run Tensorboard
# tensorboard --logdir=logs/
