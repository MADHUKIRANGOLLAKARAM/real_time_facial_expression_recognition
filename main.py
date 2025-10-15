import cv2                        # image and video processing
import streamlit as st             #used to build interactive web pages
from deepface import DeepFace      #facial recognition and analysis
from PIL import Image               # used for image modification
import os                           # operating system
import time                        # for some time
import pandas as pd                # for importing dataset
import numpy as np               # for numerical computations
import mediapipe as mp
import base64                       # for encoding binary text to ascii text
import io
import warnings
warnings.filterwarnings("ignore")


# Add custom CSS for full page background color and table size
st.markdown(
    """
    <style>
        body {
            background-color: #e0f7fa;
            color: #333333;
            font-family: 'Arial', sans-serif;
        }
        .streamlit-expanderHeader {
            background-color: #0288d1;
            color: white;
        }
        .css-1v3fvcr {
            background-color: #f0f0f5;
        }
        h1, h2, h3, p {
            text-align: center;
        }
        .stButton>button {
            background-color: #0288d1;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
        }
        .stButton>button:hover {
            background-color: #01579b;
        }
        .stImage {
            display: block;
            margin: 0 auto;
        }
        /* Increase table size and font */
        .dataframe {
            width: 100% !important;
            font-size: 20px !important;
        }
        .dataframe table {
            font-size: 20px !important;  /* Increase font size in table cells */
            width: 100% !important; /* Ensure table fills the container */
            margin-left: auto;
            margin-right: auto;
        }
        /* Increase the width of the table container */
        .stDataFrame {
            width: 80% !important;  /* Reduce width to see the effect of centering */
            max-width: none !important; /* Remove any max-width limitations */
            margin-left: auto;
            margin-right: auto;
        }
        .big-font {
            font-size: 24px !important;
            text-align: center;  /* Center the person ID heading */
            display: block;
        }

        /* CSS to set column widths explicitly */
        .streamlit .dataframe table tr th:first-child {width: 50% !important;}
        .streamlit .dataframe table tr td:first-child {width: 50% !important;}
        .streamlit .dataframe table tr th:nth-child(2) {width: 50% !important;}
        .streamlit .dataframe table tr td:nth-child(2) {width: 50% !important;} 
    </style>
    """,
    unsafe_allow_html=True
)

# Function to convert image to base64 (for CSS)
def image_to_base64(image: Image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return base64.b64encode(img_byte_arr).decode('utf-8')

# System Title
st.title("Real-time Facial Recognition and Emotion Detection System")
main_dataset_directory = 'train'
if not os.path.exists(main_dataset_directory):
    st.error(f"Data Is done in {main_dataset_directory}")
else:
    emotion_folders = [f for f in os.listdir(main_dataset_directory) if os.path.isdir(os.path.join(main_dataset_directory, f)) and not f.startswith('.')]

    if not emotion_folders:
        st.warning(f"No emotion folders found in: {main_dataset_directory}")
    else:
        emotion = st.text_input("Enter the emotion to filter images:").lower().strip()

        if not emotion:
            st.info("Please enter an emotion to filter the images.")
        else:
            st.success(f"Searching for images with emotion '{emotion}'...")
            found_any = False
            displayed_count = 0  # Counter for displayed images

            for folder in emotion_folders:
                dataset_directory = os.path.join(main_dataset_directory, folder)

                if emotion in folder.lower() or any(emotion in f.lower() for f in os.listdir(dataset_directory) if os.path.isfile(os.path.join(dataset_directory, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')) and not f.startswith('.')):  # Check folder and filenames
                    found_any = True
                    st.subheader(f"Images matching '{emotion}' in folder: '{folder}'")

                    image_files = [
                        f for f in os.listdir(dataset_directory)
                        if os.path.isfile(os.path.join(dataset_directory, f))
                        and f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))
                        and not f.startswith('.')
                        and emotion in f.lower() or emotion in folder.lower()  # Filter for file names and folder name
                    ]

                    # Use Streamlit columns to display images side by side
                    cols = st.columns(3)  # Three columns for side-by-side display
                    for idx, img_name in enumerate(image_files):
                        if displayed_count < 5:  # Limit to 5 images
                            image_path = os.path.join(dataset_directory, img_name)
                            try:
                                image = Image.open(image_path)
                                # Display images in columns with medium size (width 300px) and CSS border
                                cols[idx % 3].markdown(
                                    f'<div style="text-align:center; padding: 2px; border: 2px solid #0288d1; margin: 2px;">'
                                    f'<img src="data:image/png;base64,{image_to_base64(image)}" width="300" />'
                                    f'</div>', unsafe_allow_html=True
                                )
                                displayed_count += 1
                            except Exception as e:
                                st.error(f"Error displaying image {img_name}: {e}")
                        else:
                            break  # Exit inner loop if 5 images are displayed
                if displayed_count >= 5:
                    break  # Exit outer loop if 5 images are displayed
            if not found_any:
                st.error(f"No images found for the emotion '{emotion}' in any subfolders.")


# Function for real-time webcam capture and analysis
def detect_emotions_from_webcam(duration=30):
    """
    Captures video from the webcam, detects faces, analyzes emotions,
    and displays results in separate tables for each detected person.

    Args:
        duration (int): The duration of the capture in seconds.
    """
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Failed to access the webcam.")
        return

    video_placeholder = st.empty()
    start_time = time.time()
    all_emotion_data = {}  # Store emotion data for each face ID (person)

    face_id_counter = 0
    face_id_map = {}

    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(image_rgb)

        if results.detections:
            for i, detection in enumerate(results.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                try:
                    face_roi = frame[y:y + h, x:x + w]
                    if face_roi.size == 0:
                        print("Empty face_roi after extraction.")
                        continue
                except Exception as e:
                    st.error(f"Error extracting face ROI: {e}")
                    continue

                try:
                    result = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)
                    if result and result[0] and 'dominant_emotion' in result[0]:
                        emotion = result[0]["dominant_emotion"]
                        confidence = result[0]["emotion"][emotion]  # Get confidence for dominant emotion

                        # Assign a face ID (crude implementation - can improve)
                        if i not in face_id_map:
                            face_id_counter += 1
                            face_id_map[i] = face_id_counter
                        face_id = face_id_map[i]

                        # Display Emotion and confidence on boundary box.
                        cv2.putText(frame, f"Person {face_id}: {emotion} ({confidence:.2f}%)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # Store emotion data for this face ID
                        if face_id not in all_emotion_data:
                            all_emotion_data[face_id] = []
                        all_emotion_data[face_id].append({"Emotion": emotion, "Confidence": confidence})

                except Exception as e:
                    print(f"Error during DeepFace analysis: {e}")
                    continue
        else:
            print("No faces detected in this frame.")

        # Display the live video
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()

    # Display emotion data in separate tables for each person
    for face_id, emotion_data in all_emotion_data.items():
        st.markdown(f'<p class="big-font">Detected Emotions for Person {face_id}:</p>', unsafe_allow_html=True)
        if emotion_data:
            df = pd.DataFrame(emotion_data)
            df['Confidence'] = df['Confidence'].map('{:.2f}%'.format)  # Formats confidence as percentage string
            st.dataframe(df.style.set_properties(**{'text-align': 'center'})) # Center table content
        else:
            st.warning(f"No emotions detected for Person {face_id}.")

# Streamlit UI
st.write("Click the button below to analyze emotions live for 30 seconds.")

if st.button("Analyze Live Emotions"):
    detect_emotions_from_webcam()