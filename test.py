import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

# Constants
MODEL_PATH = 'emotion_model.h5'
LABEL_ENCODER_PATH = 'label_encoder.npy'

def extract_features(file_path):
    """
    Extracts MFCC features from the audio file for testing.
    """
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

def predict_emotion(audio_path):
    """
    Predicts the emotion of a given audio file.
    """
    # Load the trained model
    model = load_model(MODEL_PATH)
    
    # Load the label encoder
    label_classes = np.load(LABEL_ENCODER_PATH, allow_pickle=True)
    label_encoder = LabelEncoder()
    label_encoder.classes_ = label_classes

    # Extract features from the audio file
    features = extract_features(audio_path)
    
    # Reshape the features to match the model input
    features = features.reshape(1, features.shape[0], 1, 1)
    
    # Predict the emotion
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    
    return predicted_label[0], np.max(prediction)  # Emotion and confidence

if __name__ == "__main__":
    # Get the file path from the user
    audio_file = input("Enter the path to the audio file for emotion recognition: ")

    if not os.path.exists(audio_file):
        print(f"File {audio_file} not found. Please provide a valid file path.")
    else:
        emotion, confidence = predict_emotion(audio_file)
        print(f"Predicted Emotion: {emotion}")
        print(f"Confidence: {confidence * 100:.2f}%")
    