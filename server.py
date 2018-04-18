import os
from PIL import Image
from skimage.io import imsave
import numpy as np
import random
from keras import optimizers
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.layers import InputLayer , Conv2D ,MaxPooling2D ,Dense , UpSampling2D
from keras.preprocessing.image import img_to_array, load_img, ImageDataGenerator
from IPython.display import display, Image
from skimage.color import rgb2lab , lab2rgb ,rgb2gray
from flask import Flask, render_template, request , send_from_directory , redirect ,url_for


app = Flask(__name__)

UPLOAD_FOLDER = os.path.basename('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def hello_world():
    return render_template('in.html')

@app.route('/loaderio-8b3d92bd6d21792a3c4cc5fc46eda5a2.html')
def verify():
    return render_template('loaderio-8b3d92bd6d21792a3c4cc5fc46eda5a2.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    l = "world"
    
    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f)
     
    from keras.models import model_from_json
    model_file = open('model/model.json','r')
    loaded_model = model_file.read()
    model_file.close()
    model = model_from_json(loaded_model)
    model.load_weights('model/model.h5')    

    image = img_to_array(load_img('static/'+file.filename))
    image = np.array(image, dtype=float)

    X = rgb2lab(1.0/255*image)[:,:,0]
    Y = rgb2lab(1.0/255*image)[:,:,1:]
    Y /= 128
    X = X.reshape(1, len(image), len(image[0]), 1)
    Y = Y.reshape(1, len(image), len(image[0]), 2)

    output = model.predict(X)

    new_file_name = random.randint(1,1000)
    
    output *= 128
    cur = np.zeros((len(image), len(image[0]), 3))
    cur[:,:,0] = X[0][:,:,0]
    cur[:,:,1:] = output[0]
    imsave("static/"+str(new_file_name)+".png", lab2rgb(cur))
    imsave("static/"+str(new_file_name)+"_grey_version.png", rgb2gray(lab2rgb(cur)))



    return render_template('out.html',variable=file.filename+"123",v1="static/"+str(new_file_name)+"_grey_version.png",v2="static/"+str(new_file_name)+".png")

@app.errorhandler(500)
def internal_server_error(error):
    return redirect(url_for('hello_world'))
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__ == '__main__':
    app.run()
