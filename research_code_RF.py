# setting up the data path
import os

os.chdir("E:/MSc/Research/satellite verification one shot/data/RF/")

# Importing all the necessary libraries
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
import os
import numpy as np


def image_to_pixel():
    os.getcwd()
    collection = 'E:/MSc/Research/satellite verification one shot/data/test case 3/train/invalid'
    results = pd.DataFrame()
    for i, filename in enumerate(os.listdir(collection)):
        filename1 = filename[:len(filename) - 4]
        img_path = 'E:/MSc/Research/satellite verification one shot/data/test case 3/train/invalid/' + filename1 + '.jpg'
        img = image.load_img(img_path, target_size=(10, 10))
        img_str = np.array(img).flatten().tolist()
        row = pd.DataFrame(img_str)
        row = row.transpose()
        results = results.append(row)
    return results

# result = image_to_pixel()
# result.to_csv('E:/MSc/Research/satellite verification one shot/data/RF/test case 3/train_invalid.csv')


# Importing Train and Test datasets
train_data = pd.read_csv("E:/MSc/Research/satellite verification one shot/data/RF/test case 3/train_set.csv")
final_test_data = pd.read_csv("E:/MSc/Research/satellite verification one shot/data/RF/test case 3/test_set.csv")

# Splitting independent variables from the dependent variable in both training and testing
X_train = train_data.iloc[:, 1:]
y_train = train_data.label.astype("str")

X_final_test = final_test_data.iloc[:, 1:]
y_final_test = final_test_data.label.astype("str")

# Splitting train data into training and validation datasets
x_train, x_test, y_train_v, y_test_v = train_test_split(X_train, y_train, test_size=0.3, random_state=2)


# ================== Using Random Forest without hyper paramter tuning ===================
rf = RandomForestClassifier()

rf.fit(x_train, y_train_v)

# Predictions on training and validation
y_pred_train = rf.predict(x_train)

# predictions for test
y_pred_test = rf.predict(x_test)

# training metrics
# print("Training metrics:")
# print(sklearn.metrics.classification_report(y_true=y_train_v, y_pred=y_pred_train))

# test data metrics
# print("Test data metrics:")
# print(sklearn.metrics.classification_report(y_true=y_test_v, y_pred=y_pred_test))

# Predictions on testset
y_pred_test = rf.predict(X_final_test)


# test data metrics
print("Test data metrics:")
print(sklearn.metrics.confusion_matrix(y_final_test, y_pred_test))
print(sklearn.metrics.classification_report(y_true=y_final_test, y_pred=y_pred_test))
