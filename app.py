from __future__ import division, print_function
import sys
import os
import glob
import re
import numpy as np

from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

from flask import Flask,url_for,render_template,request
from flask_bootstrap import Bootstrap
app=Flask(__name__)

MODEL_PATH = 'C:/Users/Arudhra Narasimhan V/Desktop/FlaskDeploy/model_resnet.h5'

model = load_model(MODEL_PATH)
model._make_predict_function()

@app.route('/')
def home():
	return render_template('Retinopathy.html')

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    return preds

 @app.route('/predict,methods=['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model)       
        pred_class = decode_predictions(preds, top=1)   
        result = str(pred_class[0][1][2][3])              
        return result
    return None   	

if __name__=='__main__':
	app.run(debug=True)