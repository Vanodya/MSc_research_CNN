import streamlit as st
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import os
from PIL import Image
import glob

st.title('QC - Satellite Antenna Installation')


def get_prediction(classifier, img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    prediction = classifier.predict(images)
    predicted_class = np.argmax(prediction, axis=1)
    return predicted_class[0]


def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('E:/repos/dtv-algo_changes/dtv-algorithm/app_project/static/images/',uploaded_file.name),'wb') as f:
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0


def get_model(TestCase):
    classifier = load_model('E:/repos/dtv-algo_changes/dtv-algorithm/app_project/static/test case ' + str(TestCase) + '.h5')
    return classifier


def main():
    TestCase = st.sidebar.selectbox('Test Case No',(1,2,3))
    col1, col2 = st.columns(2)
    col1.subheader('Template')
    if TestCase is not None:
        col1.image(Image.open('E:/repos/dtv-algo_changes/dtv-algorithm/app_project/static/templates/' + str(TestCase) +
                            '.jpg'),width=200)
    image_file = st.sidebar.file_uploader("Upload Image of the Installation", type=["jpg"])
    removing_files = glob.glob('E:/repos/dtv-algo_changes/dtv-algorithm/app_project/static/images/*.jpg')
    for i in removing_files:
        os.remove(i)
    success = save_uploaded_file(image_file)
    col2.subheader('Uploaded Image')
    if image_file is not None:
        col2.image(
            Image.open('E:/repos/dtv-algo_changes/dtv-algorithm/app_project/static/images/' + image_file.name),
            width=200)

    if col2.button("Predict"):
        if success == 1:
            img_path = 'E:/repos/dtv-algo_changes/dtv-algorithm/app_project/static/images/' + image_file.name
            try:
                classifier_path = get_model(TestCase)
                prediction = get_prediction(classifier_path, img_path)
                if prediction == 1:
                    result = 'CORRECT'
                    st.success('The installation is {}'.format(result))
                if prediction == 0:
                    result = 'INCORRECT'
                    st.error('The installation is {}'.format(result))
            except:
                result = 'ERROR'
                st.error('The installation is {}'.format(result))
        else:
            col2.caption('Upload an image to proceed!')


main()

# streamlit run App_development.py
