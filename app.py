from flask import Flask, render_template, request, redirect, url_for
from implement_hog import process_image, process_video, BoxesManager
import os
import cv2

app = Flask(__name__)

result_path = None

@app.route('/')
def index():
    global result_path
    return render_template('index.html', result_path=result_path)

@app.route('/result')
def result():
    global result_path
    return render_template('result.html', result_path=result_path)

@app.route('/reset_result_path')
def reset_result_path():
    global result_path
    result_path = None
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    global result_path

    uploaded_file = request.files['file']

    if uploaded_file.filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = 'uploads/uploaded_image.jpg'
        uploaded_file.save(image_path)
        test_img = cv2.imread(image_path)
        result_image = process_image(test_img, BoxesManager(n=30))
        result_path = 'static/result_image.jpg'
        cv2.imwrite(result_path, result_image)

    elif uploaded_file.filename.endswith('.mp4'):
        video_path = 'uploads/uploaded_video.mp4'
        uploaded_file.save(video_path)
        output_video_path = 'static/result_video.mp4'
        process_video(video_path, output_video_path)
        result_path = output_video_path

    else:
        return render_template('index.html', result_path=None)

    return redirect(url_for('result'))

if __name__ == '__main__':
    app.run(debug=True)
