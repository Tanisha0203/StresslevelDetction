from flask import Flask, Response, render_template
from test2 import VideoCamera

app = Flask(__name__, template_folder=r'C:\Users\HARSH\Desktop\StresslevelDetction\templates')

@app.route('/')
def home():
    return render_template('index.html')

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        if frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        else:
            break

@app.route("/predict")
def predict():  
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, threaded=True)



# libraries : 
# flask : for frontend
#  dlib: Face detection and facial landmark prediction.
#  cv2 (OpenCV): Video capture, image processing, and display of results.
#  Keras: Emotion classification using a pre-trained model.
#  imutils: Image resizing utilities.  Resizes the input frame, ensuring the width doesn’t exceed 500 pixels while maintaining the aspect ratio.
#  scipy: Euclidean distance calculation.
#  numpy: Numerical and array operations, normalization, and scaling.
# matplotlib= Matplotlib is a plotting library, often used for visualizing data, including plotting the training and validation loss/accuracy curves of your machine learning model.
# tensorflow = TensorFlow is an open-source machine learning library,  training, and deploying deep learning models (like the one in your stress detection system)



# Model and Data:
# shape_predictor_68_face_landmarks.dat: This is a pre-trained model for detecting 68 facial landmarks. It's used with dlib's shape_predictor to predict the facial landmarks in the detected face.


# _mini_XCEPTION.102-0.66.hdf5: This is the pre-trained Keras model used for emotion classification. The model predicts emotions based on facial expressions, such as "happy", "sad", "angry", etc.