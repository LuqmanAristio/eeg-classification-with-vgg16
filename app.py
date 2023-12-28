from flask import Flask, render_template, request, redirect, url_for
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from flask import session

app = Flask(__name__)
app.secret_key = '123'

MODEL_PATH = 'model\model.h5'
CLASS_LABELS = {0: 'Eyes Open', 1: 'Eyes Closed', 2: 'Open and Close Fist'}

loaded_model = load_model(MODEL_PATH)

result_path = None

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = UPLOAD_FOLDER

def preprocess_image(img):
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.resnet50.preprocess_input(img)
    return img

@app.route('/')
def index():
    global result_path
    return render_template('index.html', result_path=result_path)

@app.route('/result')
def result():
    global result_path

    prediction_result = session.get('prediction_result', {})

    return render_template('result.html', result_path=result_path, prediction_result=prediction_result)

@app.route('/reset_result_path')
def reset_result_path():
    global result_path
    result_path = None
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    global result_path

    image_file = request.files['file']
    
    if not image_file.filename:
        return redirect(url_for('index'))

    try:
        img = Image.open(image_file.stream).convert('RGB')
        img = img.resize((224, 224))

        result_path = 'static/result_image.jpg'
        img.save(result_path)
        
        processed_image = preprocess_image(img)
        prediction = loaded_model.predict(processed_image)
        predicted_class_index = np.argmax(prediction)
        
        predicted_class_label = CLASS_LABELS.get(predicted_class_index, 'Unknown')
        
        response = {'predicted_class': predicted_class_label,
                    'confidence': float(prediction[0, predicted_class_index])}
        
        session['prediction_result'] = response
        result_path = "filled"

        return redirect(url_for('result'))
    
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
