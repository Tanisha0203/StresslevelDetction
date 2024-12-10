import dlib
import cv2
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from scipy.spatial import distance as dist
import imutils
from imutils import face_utils
import numpy as np

# Global variables and imports should be declared first
global points, points_lip, emotion_classifier, detector, predictor

# Define all helper functions first  _  eyes distance 
def ebdist(leye, reye):
    eyedist = dist.euclidean(leye, reye)
    points.append(int(eyedist))
    return eyedist

# lips distance 
def lpdist(l_lower, l_upper):
    lipdist = dist.euclidean(l_lower, l_upper)
    points_lip.append(int(lipdist))
    return lipdist

# emotion finder
def emotion_finder(faces, frame):
    EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
    x, y, w, h = face_utils.rect_to_bb(faces)
    frame = frame[y:y+h, x:x+w]
    roi = cv2.resize(frame, (64, 64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)
    preds = emotion_classifier.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    if label in ['scared', 'sad', 'angry']:
        label = 'Stressed'
    else:
        label = 'Not Stressed'
    return label

def normalize_values(points, disp, points_lip, dis_lip):
    normalize_value_lip = abs(dis_lip - np.min(points_lip)) / abs(np.max(points_lip) - np.min(points_lip))
    normalized_value_eye = abs(disp - np.min(points)) / abs(np.max(points) - np.min(points))
    normalized_value = (normalized_value_eye + normalize_value_lip) / 2
    stress_value = (np.exp(-normalized_value)) * 10  # Scale stress value from 1 to 10
    stress_value = np.clip(stress_value, 1, 10)  # Ensure stress value is between 1 and 10

    if stress_value >= 8:
        stress_label = "Low Stress"
    elif stress_value >= 6:
        stress_label = "Moderate Stress"
    else:
        stress_label = "High Stress"
    
    return stress_value, stress_label

# Load models and data
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotion_classifier = load_model("_mini_XCEPTION.102-0.66.hdf5", compile=False)
points = []
points_lip = []

# Class definition should come after function definitions
class VideoCamera(object):
    def __init__(self, video_source=r'C:\Users\HARSH\Desktop\StresslevelDetction\test.mp4'):
        # Initialize with the video file path
        self.video = cv2.VideoCapture(video_source)

    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        ret, frame = self.video.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=500, height=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detections = detector(gray, 0)
        for detection in detections:
            emotion = emotion_finder(detection, gray)
            cv2.putText(frame, emotion, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            shape = predictor(gray, detection)
            shape = face_utils.shape_to_np(shape)

            # Calculate distances for stress detection
            left_eye = shape[36:42]  # Landmark points for left eye
            right_eye = shape[42:48]  # Landmark points for right eye
            lip_lower = shape[48]  # Landmark point for lower lip
            lip_upper = shape[51]  # Landmark point for upper lip
            
            # Calculate distances
            eye_distance = ebdist(left_eye.mean(axis=0), right_eye.mean(axis=0))
            lip_distance = lpdist(lip_lower, lip_upper)

            # Normalize values and get stress value and label
            stress_value, stress_label = normalize_values(points, eye_distance, points_lip, lip_distance)

            # Display stress value and label
            cv2.putText(frame, "Stress Value: {:.1f}".format(stress_value), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "Stress Level: {}".format(stress_label), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw green lines on the eyebrows (left and right)
            left_eyebrow = shape[17:22]
            right_eyebrow = shape[22:27]
            cv2.polylines(frame, [np.int32(left_eyebrow)], isClosed=False, color=(0, 255, 0), thickness=2)
            cv2.polylines(frame, [np.int32(right_eyebrow)], isClosed=False, color=(0, 255, 0), thickness=2)

            # Draw green lines on the lips
            lips = shape[48:60]
            cv2.polylines(frame, [np.int32(lips)], isClosed=True, color=(0, 255, 0), thickness=2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
