import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import os
import json
import requests
import shutil

# Anomaly classes and emoji symbols
class_labels = [
    "cheat passing", "copying", "discussing", "normal",
    "peeking", "showing answer", "suspicious", "using copy cheat", "using mobile"
]

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

# Paths
history_file = "history1.json"
processed_videos_folder = "processed_videos/"
uploaded_videos_folder = "uploaded_videos/"
anomaly_screenshots_folder = "anomaly_screenshots/"
backend_url = "http://localhost:5000"

# Ensure directories exist
for folder in [processed_videos_folder, uploaded_videos_folder, anomaly_screenshots_folder]:
    os.makedirs(folder, exist_ok=True)


# Load/save history
def load_history_data():
    try:
        if os.path.exists(history_file) and os.path.getsize(history_file) > 0:
            with open(history_file, "r") as f:
                return json.load(f)
        return []
    except json.JSONDecodeError:
        return []


def save_history_data(data):
    with open(history_file, "w") as f:
        json.dump(data, f, indent=4)


history_data = load_history_data()


# Style
def add_background_color():
    st.markdown("""
        <style>
        .stApp {
            background-color: #FFDEE9;
            background-image: linear-gradient(to top, #f3e7e9 0%, #e3eeff 99%, #e3eeff 100%);
            min-height: 100vh;
        }
        </style>
    """, unsafe_allow_html=True)


# Set config
st.set_page_config(page_title="Anomaly Detection System", layout="wide", page_icon="🎥")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page:", ["Home", "Detection", "History"],
                        format_func=lambda x: {
                            "Home": "🏠 Home",
                            "Detection": "🔎 Detection",
                            "History": "🕒 History",
                        }.get(x, x)
                        )

# Home
if page == "Home":
    st.title("🎥 Anomaly Detection System for Offline Exam Monitoring")
    add_background_color()
    st.write("## Welcome to the Anomaly Detection System!")
    st.markdown("- 📽 Upload a video for automated anomaly detection.")
    st.markdown("- 🧠 Powered by CNNs trained on TensorFlow and Keras.")
    st.markdown("- ⚡ Real-time frame analysis and annotation.")

# Detection
elif page == "Detection":
    st.title("🔎 Test Video for Anomaly Detection")
    add_background_color()
    st.write("## 📽 Upload a video file")

    session_id = datetime.now().strftime('%Y%m%d%H%M%S')
    uploaded_file = st.file_uploader("Upload a video:", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file:
        uploaded_video_path = os.path.join(uploaded_videos_folder, f"uploaded_video_{session_id}.mp4")
        with open(uploaded_video_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("⏳ Processing video..."):
            # Check backend connectivity
            try:
                requests.get(backend_url, timeout=3)
            except requests.exceptions.RequestException:
                st.error("❌ Backend server not reachable. Make sure it's running at http://localhost:5000.")
                st.stop()

            try:
                with open(uploaded_video_path, 'rb') as video_file:
                    files = {'video': video_file}
                    response = requests.post(f"{backend_url}/upload_video", files=files)

                if response.status_code == 200:
                    result = response.json()
                    output_video_path = result.get("output_video")
                    anomalies = result.get("anomalies", [])
                    screenshot_paths = result.get("screenshot_paths", [])

                    if not output_video_path or not anomalies:
                        st.error("Missing output video or anomaly data from backend.")
                        st.stop()

                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    history_data.append({
                        "timestamp": timestamp,
                        "video_name": uploaded_file.name,
                        "anomalies": anomalies,
                        "output_video": output_video_path,
                        "screenshot_paths": screenshot_paths
                    })
                    save_history_data(history_data)

                    st.success("✅ Video processing completed!")
                    st.write(f"### Detected Anomalies: {', '.join(anomalies)}")
                    st.video(output_video_path)

                    st.write("### Anomaly Screenshots folder:")
                    if screenshot_paths:
                        video_id = screenshot_paths[0].split("/")[0]
                        zip_url = f"{backend_url}/download_screenshots/{video_id}"
                        st.markdown(f"📁 [Download Anomaly Screenshots as ZIP]({zip_url})", unsafe_allow_html=True)
                    else:
                        st.warning("No screenshots available.")

                else:
                    st.error(f"Error from backend: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f" An unexpected error occurred: {e}")

    st.markdown("---")

    # Webcam stream
    st.write("## 🎥 Real-Time")
    st.write("Click the button below to open the webcam stream in a new tab:")

    if st.button("Open Real-Time Webcam Stream"):
        st.components.v1.html("""<iframe src="http://localhost:5000/stream" width="1000" height="700"></iframe>""",
                              height=600)

    if st.button("Stop Real-Time Webcam Stream "):
        try:
            res = requests.get(f"{backend_url}/stop_stream")
            data = res.json()

            if res.status_code == 200 and "output_video" in data:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                session_id = datetime.now().strftime('%Y%m%d%H%M%S')

                if res.status_code == 200 and "output_video" in data:
                    # Save output video path
                    history_data.append({
                        "timestamp": timestamp,
                        "session_id": session_id,
                        "video_name": f"cctv_{session_id}",  # Video name as expected
                        "anomalies": data.get("anomalies", []),
                        "output_video": data["output_video"],  # Path returned from backend
                        "screenshot_paths": [f"{os.path.basename(os.path.dirname(data['output_video']))}/{screenshot}"
                                             for screenshot in data.get("screenshot_paths", [])]
                    })

                save_history_data(history_data)  # Save history data to file

                st.success("✅ Real-time stream stopped and saved to history.")
            else:
                st.warning("⚠️ Could not save real-time stream to history. Backend response incomplete.")
        except Exception as e:
            st.error(f"Error stopping stream: {e}")


# History
elif page == "History":
    st.title("🕒 History")
    add_background_color()

    if not history_data:
        st.info("No history available.")
    else:
        for entry in reversed(history_data):
            # Display video name and timestamp
            st.write(f"**Video**: {entry['video_name']} - **Timestamp**: {entry['timestamp']}")
            st.write(f"**Anomalies**: {', '.join(entry['anomalies']) if entry['anomalies'] else 'None'}")

            # Check if output video exists, and display or show error
            if entry.get("output_video") and os.path.exists(entry["output_video"]):
                st.video(entry["output_video"])  # Display video using Streamlit
            else:
                st.error(f" Output video not found: {entry.get('output_video', 'Unknown path')}")

            # Anomaly screenshots section
            st.write("### Anomaly Screenshots folder:")
            if entry.get("screenshot_paths"):
                video_id = entry["screenshot_paths"][0].split("/")[0]
                zip_url = f"{backend_url}/download_screenshots/{video_id}"
                st.markdown(f"📁 [Download Anomaly Screenshots as ZIP]({zip_url})", unsafe_allow_html=True)
            else:
                st.warning("No screenshots available.")

            # Button to delete video from history and filesystem
            if st.button(f"Delete {entry['video_name']}", key=entry['timestamp']):
                history_data.remove(entry)  # Remove from history
                try:
                    # Delete the output video file
                    if os.path.exists(entry["output_video"]):
                        os.remove(entry["output_video"])
                        st.success(f"✅ Deleted video {entry['output_video']} from filesystem.")
                    else:
                        st.warning(f"⚠️ Video not found: {entry['output_video']}")

                except Exception as e:
                    st.warning(f"Could not delete video: {e}")

                try:
                    # Delete screenshot folder if it exists
                    if entry.get("screenshot_paths"):
                        folder = entry["screenshot_paths"][0].split("/")[0]
                        screenshot_dir = os.path.join(anomaly_screenshots_folder, folder)
                        if os.path.exists(screenshot_dir):
                            shutil.rmtree(screenshot_dir)
                            st.success(f"✅ Deleted screenshot folder {screenshot_dir}.")
                except Exception as e:
                    st.warning(f"Could not delete screenshots: {e}")

                # Save updated history
                save_history_data(history_data)
                st.success(f"✅ Deleted {entry['video_name']} from history.")

            st.markdown("---")  # Separator between entries

# Footer
st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px 0;
        background-image: linear-gradient(to top, #f3e7e9 0%, #e3eeff 99%);
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
""", unsafe_allow_html=True)
