import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os
import json
from datetime import datetime

# Load the trained model
model = load_model(r"C:\Users\HP\Desktop\ADAM\improved_anomaly_model.h5")
#model = load_model(r"C:\Users\HP\Desktop\MP\anomaly_detection_cnn.h5")

# Define the class labels
class_labels = [
    "cheat passing", "copying", "discussing", "normal",
    "peeking", "showing answer", "suspicious", "using copy cheat", "using mobile"
]

# Define the corresponding emoji symbols for each class
symbols = {
    "cheat passing": "📄",
    "copying": "✍️",
    "discussing": "💬",
    "normal": "🙂",
    "peeking": "👀",
    "showing answer": "📌",
    "suspicious": "⚠️",
    "using copy cheat": "📄️",
    "using mobile": "📱",
}

# File to store history data
history_file = "history.json"
processed_videos_folder = "data/processed_videos/"
uploaded_videos_folder = "data/uploaded_videos/"

# Create the processed and uploaded videos folders if they don't exist
if not os.path.exists(processed_videos_folder):
    os.makedirs(processed_videos_folder)

if not os.path.exists(uploaded_videos_folder):
    os.makedirs(uploaded_videos_folder)

# Load or initialize history data
try:
    if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
        with open(history_file, "r") as f:
            history_data = json.load(f)
    else:
        history_data = []
except json.JSONDecodeError:
    # If the file is invalid, reset history_data and overwrite the file
    history_data = []
    with open(history_file, "w") as f:
        json.dump(history_data, f)


# Save history data
def save_history_data(history_data):
    with open(history_file, "w") as f:
        json.dump(history_data, f, indent=4)


# Add background color
def add_background_color():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-color: #FFDEE9;
                background-image: linear-gradient(to top, #f3e7e9 0%, #e3eeff 99%, #e3eeff 100%);
                min-height: 100vh;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Function to predict a frame
def predict_frame(frame):
    frame_size = (224,224)

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize the grayscale frame
    resized_frame = cv2.resize(gray_frame, frame_size)

    # Normalize the pixel values
    normalized_frame = resized_frame / 255.0

    # Add batch and channel dimensions
    input_frame = np.expand_dims(normalized_frame, axis=(0, -1))

    # Convert to RGB if the model expects 3 channels
    if model.input_shape[-1] == 3:
        input_frame = np.repeat(input_frame, 3, axis=-1)

    # Predict using the model
    predictions = model.predict(input_frame)
    predicted_class = np.argmax(predictions[0])
    predicted_label = class_labels[predicted_class]
    confidence = np.max(predictions[0])
    return predicted_label, confidence

# Function to process the video
def process_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error("Cannot open video file.")
        return None, []

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    display_width, display_height = 1920, 1080
    output_video_name = f"processed_{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4v"
    output_video_path = os.path.join(processed_videos_folder, output_video_name)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (display_width, display_height))

    anomalies_detected = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Fixed text size and thickness
    fixed_text_size = 30  # Fixed text size (30px)
    font_thickness = 3  # Fixed font thickness

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        progress_percentage = current_frame / total_frames
        progress_bar.progress(progress_percentage)

        # Status updates based on the current frame position
        if current_frame == 1:
            status_text.text("📹 Processing video... Reading frames...")
        elif current_frame > 1 and current_frame < total_frames / 3:
            status_text.text(f"🔄 Processing: {progress_percentage * 100:.2f}% - Converting frames...")
        elif current_frame > 2 * total_frames / 3:
            status_text.text(f"🔍 Processing: {progress_percentage * 100:.2f}% - Detecting anomalies...")

        try:
            predicted_label, confidence = predict_frame(frame)
        except Exception as e:
            st.error(f"Error during frame prediction: {e}")
            break

        text = f" {predicted_label} ({confidence * 100:.2f}%)"

        # Calculate the position for the text, centered at the top of the frame
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, font_thickness)
        text_x, text_y = (frame.shape[1] - text_width) // 2, 50

        # Draw rectangle for text background
        background_color = (0, 0, 0)  # Black background
        text_color = (0, 255, 255)  # Yellow text
        cv2.rectangle(frame, (text_x - 10, text_y - text_height - 10), (text_x + text_width + 10, text_y + 10),
                      background_color, -1)

        # Add text with fixed size and thickness
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, font_thickness)


        # If anomaly is detected, draw the green rectangle
        if predicted_label != "normal":
            anomalies_detected.append(predicted_label)
            play_button_y = int(frame_height * 0.1)  # Position slightly below the play button
            cv2.rectangle(frame,
                          (20, play_button_y),  # Top-left corner
                          (frame_width - 20, frame_height - 100),  # Bottom-right corner, leaving space at the bottom
                          (0, 255, 0),  # Green color
                          5)  # Thickness

            # Add bottom padding for spacing
            bottom_padding = 100  # Increased padding for better visibility
            frame_with_padding = frame.copy()  # Use a copy of the frame to avoid cropping original content

        # Draw a green outline in the bottom padding space
            cv2.rectangle(frame_with_padding,
                      (20, frame_height - bottom_padding),  # Top-left corner of bottom border
                      (frame_width - 20, frame_height - 100),  # Bottom-right corner of bottom border
                      (0, 255, 0),  # Green color
                      5)  # Thickness of the bottom border

        # Resize the frame to fit display dimensions
            resized_frame = cv2.resize(frame_with_padding, (display_width, display_height))
            out.write(resized_frame)

    cap.release()
    out.release()
    return output_video_path, list(set(anomalies_detected))


# Streamlit App
st.set_page_config(page_title="Anomaly Detection System", layout="wide", page_icon="🎥")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page:",
    ["Home", "Detection", "History", "About"],
    format_func=lambda x: {
        "Home": "🏠 Home",
        "Detection": "🔎 Detection",
        "History": "🕒 History",
        "About": "💡 About"
    }.get(x, x),
)

if page == "Home":
    st.title("🎥 Anomaly Detection System for Offline Exam Monitoring")
    add_background_color()
    st.write("## Welcome to the Anomaly Detection System!")
    st.markdown("- 📽 Upload a video for automated anomaly detection.")
    st.markdown("- 🧠 Powered by CNNs trained on TensorFlow and Keras.")
    st.markdown("- ⚡ Real-time frame analysis and annotation.")

elif page == "Detection":
    st.title("🔎 Testing Video for Anomaly Detection")
    add_background_color()
    st.write("### 📽 Upload a video file for testing anomalies in exam monitoring")

    # Create a unique upload folder with a timestamp
    session_id = str(datetime.now().strftime('%Y%m%d%H%M%S'))
    upload_folder = uploaded_videos_folder  # Set the folder for uploaded videos
    os.makedirs(upload_folder, exist_ok=True)

    uploaded_file = st.file_uploader("Upload a video file:", type=["mp4", "mp4v", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        uploaded_video_path = os.path.join(upload_folder, f"uploaded_video_{session_id}.mp4v")
        with open(uploaded_video_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("⏳ Processing video..."):
            output_video_path, anomalies = process_video(uploaded_video_path)

        if output_video_path:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            history_data.append({
                "timestamp": timestamp,
                "video_name": uploaded_file.name,
                "anomalies": anomalies,
                "output_video": output_video_path
            })
            save_history_data(history_data)
            st.success("✅ Video processing completed!")
            st.write(f"### Detected Anomalies: {', '.join(anomalies)}")
            st.video(output_video_path)

elif page == "History":
    st.title("🕒 History")
    add_background_color()
    if not history_data:
        st.info("No history available.")
    else:
        for entry in reversed(history_data):
            st.write(f"**Video**: {entry['video_name']} - **Timestamp**: {entry['timestamp']}")
            st.write(f"**Anomalies**: {', '.join(entry['anomalies'])}")
            st.video(entry["output_video"])

            # Deletion option
            if st.button(f"Delete {entry['video_name']}", key=entry['timestamp']):
                history_data.remove(entry)
                os.remove(entry["output_video"])  # Delete the video file
                save_history_data(history_data)
                st.success(f"Deleted {entry['video_name']} and its history.")
            st.markdown("---")

elif page == "About":
    st.title("💡 About the Anomaly Detection System")
    add_background_color()
    st.markdown("- **📒 Model** : Convolutional Neural Network (CNN), OpenCV")
    st.markdown("- **⚙️ Frameworks** : TensorFlow, Keras")
    st.markdown("- **🎥Input** : Pre-processed video frames resized to 224x224 pixels.")
    st.markdown("- **✅ Output** : Annotations for detected anomalies with confidence scores.")

# Footer
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px 0;
        background-image: linear-gradient(to top, #f3e7e9 0%, #e3eeff 99%, #e3eeff 100%);
        color: #2c3e50;
        font-size: 16px;
        box-shadow: 0px -2px 10px rgba(0, 0, 0, 0.1);
    }
    .footer a {
        color: #3498db;
        text-decoration: none;
    }
    </style>
    <div class="footer">
        👨‍💻 Developed for Anomaly Detection
    </div>
    """,
    unsafe_allow_html=True,
)
