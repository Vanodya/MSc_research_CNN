# rename images

# import os
# os.getcwd()
# collection = "E:/MSc/Research/satellite verification one shot/data/test images/train/invalid"
# for i, filename in enumerate(os.listdir(collection)):
#     os.rename("E:/MSc/Research/satellite verification one shot/data/test images/train/invalid/" + filename,
#               "E:/MSc/Research/satellite verification one shot/data/test images/train/invalid/" + "CAR_"
#               + filename + ".jpg")


# Reference Code
# https://paperswithcode.com/paper/signet-convolutional-siamese-network-for


import sys
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
# % matplotlib inline

import cv2
import time
import itertools
import random

from sklearn.utils import shuffle
# import tensorflow.keras as keras
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate, Dropout
from keras.models import Model

from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform

from keras.engine.topology import Layer
from keras.regularizers import l2
from keras import backend as K
# from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix


def visualize_sample_image():
    '''Function to randomly select a satellite from train set and
    print two genuine copies and one forged copy'''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
    k = np.random.randint(len(orig_train))
    orig_img_names = random.sample(orig_train[k], 2)
    forg_img_name = random.sample(forg_train[k], 1)
    orig_img1 = cv2.imread(orig_img_names[0], 0)
    orig_img2 = cv2.imread(orig_img_names[1], 0)
    forg_img = plt.imread(forg_img_name[0], 0)
    orig_img1 = cv2.resize(orig_img1, (img_w, img_h))
    orig_img2 = cv2.resize(orig_img2, (img_w, img_h))
    forg_img = cv2.resize(forg_img, (img_w, img_h))

    ax1.imshow(orig_img1, cmap='gray')
    ax2.imshow(orig_img2, cmap='gray')
    ax3.imshow(forg_img, cmap='gray')

    ax1.set_title('Valid Copy')
    ax1.axis('off')
    ax2.set_title('Valid Copy')
    ax2.axis('off')
    ax3.set_title('Invalid Copy')
    ax3.axis('off')


def generate_batch(orig_groups, forg_groups, batch_size=14):
    '''Function to generate a batch of data with batch_size number of data points
    Half of the data points will be Valid-Valid pairs and half will be Valid-Invalid pairs'''
    while True:
        orig_pairs = []
        forg_pairs = []
        gen_gen_labels = []
        gen_for_labels = []
        all_pairs = []
        all_labels = []

        for orig, forg in zip(orig_groups, forg_groups):
            # orig_pairs.extend(list(itertools.combinations(orig, 2))) # create all possible combinations of 2 images
            for i in range(len(orig)):
                orig_pairs.extend(list(itertools.product(orig[i:i + 1], random.sample(orig, 1))))
            for i in range(len(forg)):
                forg_pairs.extend(list(itertools.product(orig[i:i + 1], random.sample(forg, 1))))

        # Label for Valid-Valid pairs is 1
        # Label for Valid-Invalid pairs is 0
        gen_gen_labels = [1] * len(orig_pairs)
        gen_for_labels = [0] * len(forg_pairs)

        # Concatenate all the pairs together along with their labels and shuffle them
        all_pairs = orig_pairs + forg_pairs
        all_labels = gen_gen_labels + gen_for_labels
        del orig_pairs, forg_pairs, gen_gen_labels, gen_for_labels
        all_pairs, all_labels = shuffle(all_pairs, all_labels)

        # Note the lists above contain only the image names and
        # actual images are loaded and yielded below in batches
        # Below we prepare a batch of data points and yield the batch
        # In each batch we load "batch_size" number of image pairs
        # These images are then removed from the original set so that
        # they are not added again in the next batch.

        k = 0
        pairs = [np.zeros((batch_size, img_h, img_w, 1)) for i in range(2)]
        targets = np.zeros((batch_size,))
        for ix, pair in enumerate(all_pairs):
            img1 = cv2.imread(pair[0], 0)
            img2 = cv2.imread(pair[1], 0)
            img1 = cv2.resize(img1, (img_w, img_h))
            img2 = cv2.resize(img2, (img_w, img_h))
            img1 = np.array(img1, dtype=np.float64)
            img2 = np.array(img2, dtype=np.float64)
            img1 /= 255
            img2 /= 255
            img1 = img1[..., np.newaxis]
            img2 = img2[..., np.newaxis]
            pairs[0][k, :, :, :] = img1
            pairs[1][k, :, :, :] = img2
            targets[k] = all_labels[ix]
            k += 1
            if k == batch_size:
                yield pairs, targets
                k = 0
                pairs = [np.zeros((batch_size, img_h, img_w, 1)) for i in range(2)]
                targets = np.zeros((batch_size,))


def euclidean_distance(vects):
    '''Compute Euclidean Distance between two vectors'''
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_base_network_signet(input_shape):
    '''Base Siamese Network'''

    seq = Sequential()
    seq.add(Conv2D(96, kernel_size=(11, 11), activation='relu', name='conv1_1', strides=4, input_shape=input_shape,
                   kernel_initializer='glorot_uniform'))
    seq.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(ZeroPadding2D((2, 2)))

    seq.add(Conv2D(256, kernel_size=(5, 5), activation='relu', name='conv2_1', strides=1, kernel_initializer='glorot_uniform'))
    seq.add(BatchNormalization(epsilon=1e-06, axis=1, momentum=0.9))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(Dropout(0.3))  # added extra
    seq.add(ZeroPadding2D((1, 1)))

    seq.add(Conv2D(384, kernel_size=(3, 3), activation='relu', name='conv3_1', strides=1, kernel_initializer='glorot_uniform'))
    seq.add(ZeroPadding2D((1, 1)))

    seq.add(Conv2D(256, kernel_size=(3, 3), activation='relu', name='conv3_2', strides=1, kernel_initializer='glorot_uniform'))
    seq.add(MaxPooling2D((3, 3), strides=(2, 2)))
    seq.add(Dropout(0.3))  # added extra
    seq.add(Flatten(name='flatten'))
    seq.add(Dense(1024, activation='relu', kernel_initializer='glorot_uniform'))
    seq.add(Dropout(0.5))

    seq.add(Dense(128, activation='relu', kernel_initializer='glorot_uniform'))  # softmax changed to relu
    return seq


def compute_accuracy_roc(predictions, labels):
  '''Compute ROC accuracy with a range of thresholds on distances.
  '''
  dmax = np.max(predictions)
  dmin = np.min(predictions)
  nsame = np.sum(labels == 1)
  ndiff = np.sum(labels == 0)

  step = 0.01
  max_acc = 0
  best_thresh = -1

  for d in np.arange(dmin, dmax+step, step):
      idx1 = predictions.ravel() <= d
      idx2 = predictions.ravel() > d

      tpr = float(np.sum(labels[idx1] == 1)) / nsame
      tnr = float(np.sum(labels[idx2] == 0)) / ndiff
      acc = 0.5 * (tpr + tnr)
#       print ('ROC', acc, tpr, tnr)

      if (acc > max_acc):
          max_acc, best_thresh = acc, d

  return max_acc, best_thresh


def predict_score():
  '''Predict distance score and classify test images as Valid or Invalid'''
  test_point, test_label = next(test_gen)
  img1, img2 = test_point[0], test_point[1]

  result = model.predict([img1, img2])
  diff = result[0][0]
  print("Difference Score = ", diff)
  if diff > threshold:
      fig_text = "Its a Invalid Installation"
  else:
      fig_text = "Its a Valid Installation"

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
  ax1.imshow(np.squeeze(img1), cmap='gray')
  ax2.imshow(np.squeeze(img2), cmap='gray')
  ax1.set_title('Valid')
  if test_label == 1:
      ax2.set_title('Valid')
  else:
      ax2.set_title('Invalid')
  ax1.axis('off')
  ax2.axis('off')
  plt.figtext(.13, .8, fig_text + "\n\nDifference = " + str(diff))
  plt.show()


def compute_confusion_matrix(predicted_classes, actual_classes):
    # Confution Matrix and Classification Report
    print('Confusion Matrix')
    conf = confusion_matrix(actual_classes, predicted_classes)
    print(conf)
    print('Classification Report')
    report = classification_report(actual_classes, predicted_classes) #, target_names=actual_class_labels)
    print(report)

# ____________________________________________________________________________________________________________________#
# main


path = "E:/MSc/Research/satellite verification one shot/data/test images/"

# Get the list of all directories and sort them
dir_list = os.listdir(path)
sub_dir_list = os.listdir(path + dir_list[0])
# dir_list.sort()

# Valid satellites are stored in the list "orig_groups"
# Invalid satellites are stored in the list "forged_groups"
orig_train, forg_train, orig_test, forg_test, orig_val, forg_val = [], [], [], [], [], []
for directory in dir_list:
    for sub_directory in sub_dir_list:
        images = os.listdir(path + directory + '/' + sub_directory)
        images.sort()
        images = [path + directory + '/' + sub_directory + '/' + x for x in images]
        if directory == 'train' and sub_directory == 'valid':
            orig_train.append(images)  # train - valid
        if directory == 'train' and sub_directory == 'invalid':
            forg_train.append(images)  # train - invalid
        if directory == 'test' and sub_directory == 'valid':
            orig_test.append(images)  # test - valid
        if directory == 'test' and sub_directory == 'invalid':
            forg_test.append(images)  # test - invalid
        if directory == 'validation' and sub_directory == 'valid':
            orig_val.append(images)  # validation - valid
        if directory == 'validation' and sub_directory == 'invalid':
            forg_val.append(images)  # validation - invalid


print(len(orig_train[0]), len(forg_train[0]))
print(len(orig_test[0]), len(forg_test[0]))
print(len(orig_val[0]), len(forg_val[0]))

# All the images will be converted to the same size before processing
img_h, img_w = 155, 220


visualize_sample_image()


input_shape = (img_h, img_w, 1)

# network definition
base_network = create_base_network_signet(input_shape)

input_a = Input(shape=(input_shape))
input_b = Input(shape=(input_shape))

# because we re-use the same instance `base_network`,
# the weights of the network
# will be shared across the two branches
processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Compute the Euclidean distance between the two vectors in the latent space
distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(inputs=[input_a, input_b], outputs=distance)

batch_sz = 14
num_train_samples = 98
num_val_samples = 14
num_test_samples = 28
# num_train_samples, num_val_samples, num_test_samples

# compile model using RMSProp Optimizer and Contrastive loss function defined above
rms = RMSprop(lr=1e-4, rho=0.9, epsilon=1e-08)
model.compile(loss=contrastive_loss, optimizer=rms)

# Using Keras Callbacks, save the model after every epoch
# Reduce the learning rate by a factor of 0.1 if the validation loss does not improve for 5 epochs
# Stop the training using early stopping if the validation loss does not improve for 12 epochs
callbacks = [
  EarlyStopping(patience=12, verbose=1),
  ReduceLROnPlateau(factor=0.1, patience=5, min_lr=0.000001, verbose=1),
  ModelCheckpoint('E:/MSc/Research/satellite verification one shot/models/test_model_weights.h5', verbose=1, save_weights_only=True)
]

results = model.fit_generator(generate_batch(orig_train, forg_train, batch_sz),
                            steps_per_epoch = num_train_samples//batch_sz,
                            epochs = 2,
                            validation_data = generate_batch(orig_val, forg_val, batch_sz),
                            validation_steps = num_val_samples//batch_sz,
                            callbacks = callbacks)


model.load_weights('E:/MSc/Research/satellite verification one shot/models/test_model_weights.h5')

test_gen = generate_batch(orig_test, forg_test, batch_size= 1)
pred, tr_y = [], []
for i in range(num_test_samples):
  (img1, img2), label = next(test_gen)
  tr_y.append(label)
  pred.append(model.predict([img1, img2])[0][0])

tr_acc, threshold = compute_accuracy_roc(np.array(pred), np.array(tr_y))
print("Accuracy:  ", tr_acc, "Threshold:  ", threshold)

predict_score()


actual_classes = [int(item) for item in tr_y]
predicted_classes = [int(item > threshold) for item in pred]
compute_confusion_matrix(predicted_classes, actual_classes)


# get individual prediction -------------------------------------------------------------------------------------------
# def get_prediction(original_image, pred_image):
#   '''Predict distance score and classify the given image as Valid or Invalid'''
#
#   img1, img2 = original_image, pred_image
#
#   result = model.predict([img1, img2])
#   diff = result[0][0]
#   print("Difference Score = ", diff)
#   if diff > threshold:
#       fig_text = "Its a Invalid Installation"
#   else:
#       fig_text = "Its a Valid Installation"
#
#   fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (10, 10))
#   ax1.imshow(np.squeeze(img1), cmap='gray')
#   ax2.imshow(np.squeeze(img2), cmap='gray')
#   ax1.set_title('Valid')
#   ax2.set_title('Valid')
#   ax1.axis('off')
#   ax2.axis('off')
#   plt.figtext(.13, .8, fig_text + "\n\nDifference = " + str(diff))
#   plt.show()
#
# original_img_path = 'E:/MSc/Research/Data/test case 4/test/valid/dtv_65722186_1577074782119_10_1577075004015.jpg'
# pred_img_path = 'E:/MSc/Research/Data/test case 4/test/invalid/dtv_65772887_1583554636982_10_1583554801140.jpg'
#
# original_image = cv2.imread(original_img_path, 0)
# original_image = cv2.resize(original_image, (img_w, img_h))
# original_image = np.array(original_image, dtype=np.float64)
# original_image /= 255
# original_image = original_image[..., np.newaxis]
# fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
# ax1.set_title('Original Image')
# ax1.imshow(np.squeeze(original_image), cmap='gray')
#
# pred_image = cv2.imread(pred_img_path, 0)
# pred_image = cv2.resize(pred_image, (img_w, img_h))
# pred_image = np.array(pred_image, dtype=np.float64)
# pred_image /= 255
# pred_image = pred_image[..., np.newaxis]
# fig, (ax1) = plt.subplots(1, 1, figsize=(10, 10))
# ax1.set_title('Comparing Image')
# ax1.imshow(np.squeeze(pred_image), cmap='gray')
#
# # test_gen1 = generate_batch([[original_img_path]],[[pred_img_path]],1)
#
# num_test_samples1 = 1
# pred1, tr_y1 = [], []
#
# # for i in range(num_test_samples1):
# #   (img1, img2), label = next(test_gen1)
# #   tr_y1.append(label)
# #   pred1.append(model.predict([img1, img2])[0][0])
#
# # get_prediction(original_image, pred_image)
