#Importing some libraries 
from flask import Flask, render_template, request, send_from_directory
import cv2
import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, BatchNormalization, Flatten
import numpy as np

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#Loading h5 file containing our saved model from juptyer notebook
myModel = tf.keras.models.load_model('static/CovidModelSave.h5')

COUNT = 0

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

#render index.html as the first page 
@app.route('/')
def man():
    return render_template('index.html')

#Prediction.html will be rendered upon image upload 
@app.route('/home', methods=['POST'])
def home():
    global COUNT

    img = request.files['image']                #accesssing user uploaded file
    img.save('static/{}.jpg'.format(COUNT))             #save image within static folder    
    img_arr = cv2.imread('static/{}.jpg'.format(COUNT))             #storing image in array

    img_arr = cv2.resize(img_arr, (60,60))              #resize image to (60,60) to match our model input shape
    img_arr = img_arr / 255.0               #normalise pixel values 
    img_arr = img_arr.reshape(1,60,60,3)               

    prediction = myModel.predict(img_arr)               #Predict image class using our model


    #Store predicitons of both options (covid or no covid) in x and y
    x = round(prediction[0,0], 2)  
    y = round(prediction[0,1], 2)
    preds = np.array([x,y])
    COUNT += 1
    return render_template('prediction.html', data=preds)               #Load prediction.html with our results 
    #return render_template('prediction.html', data=prediction)


#Loads the image the user uploaded 
@app.route('/load_img')
def load_img():
    global COUNT
    return send_from_directory('static', "{}.jpg".format(COUNT-1))


if __name__ == '__main__':
    app.run(debug=True)
