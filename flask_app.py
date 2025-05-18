import os
import cv2
import numpy as np
import logging
from flask import Flask, request, jsonify, send_from_directory, send_file, Response, render_template
from tensorflow.keras.models import load_model
from datetime import datetime
import zipfile
import io
import json
from playsound import playsound
import threading
import time

app = Flask(__name__, template_folder="templates")

# Folders for videos and screenshots
UPLOAD_FOLDER = "uploaded_videos"
PROCESSED_FOLDER = "processed_videos"
SCREENSHOT_FOLDER = "anomaly_screenshots"
#CCTV_VIDEO_PATH = "http://192.168.43.133:8080/video"  #Local CCTV video for streaming
CCTV_VIDEO_PATH = "static/sample_cctv_video.mp4v"  #Local CCTV video for streaming
HISTORY_FILE= "history1.json"


# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

# Load trained model
model = load_model("multi_class_anomaly_model.h5")
class_names = [
    "cheat passing", "copying", "discussing", "normal",
    "peeking", "showing answer", "suspicious", "using copy cheat", "using mobile"
]

logging.basicConfig(level=logging.INFO)


def preprocess_frame(frame):
    resized = cv2.resize(frame, (128, 128))
    resized = resized.astype("float32") / 255.0
    reshaped = np.reshape(resized, (1, 128, 128, 3))
    return reshaped


def predict_behavior(frame):
    try:
        preprocessed = preprocess_frame(frame)
        predictions = model.predict(preprocessed, verbose=0)
        confidence = np.max(predictions)
        predicted_label = class_names[np.argmax(predictions)]
        return predicted_label, float(confidence)
    except Exception as e:
        logging.error(f"Error predicting frame: {e}")
        return "error", 0.0


def play_alert(duration=2):
    """Play an alert sound for a specified duration (in seconds)."""
    try:
        playsound("alert.mp3")  # Ensure you have an alert.mp3 file in your directory
        time.sleep(duration)  # Play for 2 seconds
    except Exception as e:
        logging.error(f"Error playing alert sound: {e}")


@app.route("/upload_video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    video = request.files["video"]
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    original_name = os.path.splitext(video.filename)[0] or "video"
    safe_name = original_name.replace(" ", "").replace(".", "")
    video_id = f"{safe_name}_{timestamp}"

    filename = f"{video_id}.mp4v"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    video.save(filepath)

    try:
        output_path, detected_behaviors, screenshot_paths = process_video(filepath, video_id)
        return jsonify({
            "message": "Video processed successfully",
            "output_video": output_path,
            "anomalies": list(set(detected_behaviors)),
            "screenshot_paths": screenshot_paths,
            "status": "completed"
        })
    except Exception as e:
        logging.error(f"Video processing error: {e}")
        return jsonify({"error": "Error processing video. Please try again."}), 500


def process_video(video_path, video_id):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(PROCESSED_FOLDER, f"processed_{video_id}.mp4v")
    detected_behaviors = []
    screenshot_paths = []

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
    frame_count = 0
    font_thickness = 3

    process_fps = 15
    frame_interval = int(fps // process_fps) if fps > process_fps else 1

    best_screenshots = {}
    video_screenshot_dir = os.path.join(SCREENSHOT_FOLDER, video_id)
    os.makedirs(video_screenshot_dir, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            label, confidence = predict_behavior(frame)

            if label != "normal" and label != "error":
                detected_behaviors.append(label)

                # Trigger sound alert if behavior is abnormal for 2 seconds
                threading.Thread( args=(2,)).start()

                if label in best_screenshots:
                    old_snapshot_path = best_screenshots[label]["path"]
                    if confidence > best_screenshots[label]["confidence"]:
                        logging.info(f"Deleting old screenshot: {old_snapshot_path}")
                        os.remove(old_snapshot_path)

                if label not in best_screenshots or confidence > best_screenshots[label]["confidence"]:
                    snapshot_filename = f"frame_{frame_count}{label.replace(' ', '')}conf{confidence:.2f}.jpg"
                    snapshot_path = os.path.join(video_screenshot_dir, snapshot_filename)

                    logging.info(f"Saving screenshot at: {snapshot_path}")
                    cv2.imwrite(snapshot_path, frame)
                    best_screenshots[label] = {
                        "path": snapshot_path,
                        "confidence": confidence,
                        "filename": snapshot_filename
                    }

            # Draw text and overlay with confidence
            text = f"{label} ({confidence:.2f})"
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, font_thickness)
            text_x, text_y = (frame.shape[1] - text_width) // 2, 50
            cv2.rectangle(frame, (text_x - 10, text_y - text_height - 10),
                          (text_x + text_width + 10, text_y + 10), (0, 0, 0), -1)
            cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), font_thickness)

            if label != "normal":
                cv2.rectangle(frame, (20, int(frame_height * 0.1)),
                              (frame_width - 20, frame_height - 100), (0, 255, 0), 5)
                cv2.rectangle(frame, (20, frame_height - 100),
                              (frame_width - 20, frame_height - 100 + 5), (0, 255, 0), 5)

        out.write(frame)
        frame_count += 1

        if frame_count % 10 == 0:
            logging.info(f"Processing frame {frame_count}/{total_frames}...")

    cap.release()
    out.release()
    os.remove(video_path)

    for info in best_screenshots.values():
        screenshot_paths.append(f"{video_id}/{info['filename']}")

    return out_path, detected_behaviors, screenshot_paths


def generate_cctv_stream():
    cap = cv2.VideoCapture(CCTV_VIDEO_PATH)
    if not cap.isOpened():
        logging.error("Could not open CCTV video")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    timestamp_display = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    video_id = f"cctv_{timestamp}"
    output_path = os.path.join(PROCESSED_FOLDER, f"{video_id}.mp4v")
    screenshot_dir = os.path.join(SCREENSHOT_FOLDER, video_id)
    os.makedirs(screenshot_dir, exist_ok=True)

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    best_screenshots = {}
    frame_count = 0
    detected_behaviors = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        label, confidence = predict_behavior(frame)

        if label != "normal" and label != "error":
            detected_behaviors.append(label)
            threading.Thread(target=play_alert, args=(2,)).start()

            if label not in best_screenshots or confidence > best_screenshots[label]["confidence"]:
                filename = f"frame_{frame_count}_{label.replace(' ', '')}_{confidence:.2f}.jpg"
                snapshot_path = os.path.join(screenshot_dir, filename)
                cv2.imwrite(snapshot_path, frame)
                best_screenshots[label] = {
                    "filename": filename,
                    "path": snapshot_path,
                    "confidence": confidence
                }

        font_thickness=3

        # Overlay prediction
        text = f"{label} ({confidence:.2f})"
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, font_thickness)
        text_x, text_y = (frame.shape[1] - text_width) // 2, 50
        cv2.rectangle(frame, (text_x - 10, text_y - text_height - 10),
                      (text_x + text_width + 10, text_y + 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), font_thickness)

        if label != "normal":
            cv2.rectangle(frame, (20, int(frame.shape[0] * 0.1)),
                          (frame.shape[1] - 20, frame.shape[0] - 100), (0, 255, 0), 4)  # green color

            # Trigger alert sound if behavior is abnormal
            threading.Thread(target=play_alert, args=(2,)).start()

        # Save to output
        out.write(frame)

        # Stream to UI
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        frame_count += 1

    cap.release()
    out.release()

    # Save history
    screenshot_files = [f"{video_id}/{info['filename']}" for info in best_screenshots.values()]
    history_entry = {
        "timestamp": timestamp_display,
        "video_name": video_id,
        "video_path": f"{video_id}.mp4v",
        "anomalies": list(set(detected_behaviors)),
        "output_video":  output_path,
        "screenshot_paths": screenshot_files,
    }

    try:
        with open(HISTORY_FILE, "r+") as f:
            data = json.load(f)
            data.append(history_entry)
            f.seek(0)
            json.dump(data, f, indent=4)
    except Exception as e:
        logging.error(f"Failed to update history file: {e}")

stream_active = True
stream_anomalies = set()
stream_screenshot_paths = []
stream_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
stream_output_path = os.path.join(PROCESSED_FOLDER, f"stream_{stream_timestamp}.mp4v")

def process_stream():
    global stream_active, stream_anomalies, stream_screenshot_paths, stream_output_path

    stream_active = True  # Reset flag
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        logging.error("Failed to open webcam.")
        return

    # Define the output file and video writer
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")  # Remove colons
    stream_output_path = os.path.join(PROCESSED_FOLDER, f"stream_{timestamp}.mp4v")
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)  # Ensure folder exists
    out = cv2.VideoWriter(stream_output_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (640, 480))

    frame_count = 0
    best_screenshots = {}

    while stream_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Anomaly detection logic
        label, confidence = predict_behavior(frame)

        # If an anomaly is detected (not "normal" and not "error")
        if label != "normal" and label != "error":
            stream_anomalies.add(label)  # Add to detected anomalies

            # Save the screenshot for this anomaly
            if label not in best_screenshots or confidence > best_screenshots[label]["confidence"]:
                # Screenshot path for anomalies
                screenshot_filename = f"frame_{frame_count}_{label.replace(' ', '')}_conf_{confidence:.2f}.jpg"
                screenshot_path = os.path.join(SCREENSHOT_FOLDER, f"{timestamp}/{screenshot_filename}")

                os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
                cv2.imwrite(screenshot_path, frame)

                best_screenshots[label] = {
                    "path": screenshot_path,
                    "confidence": confidence,
                    "filename": screenshot_filename
                }

            # Update screenshot paths for response
            stream_screenshot_paths = [info["path"] for info in best_screenshots.values()]

        # Write frame to the output video
        out.write(frame)

        # Increment frame count
        frame_count += 1

    cap.release()
    out.release()  # Ensure output file is properly closed

    logging.info("Streaming finished or stopped.")



@app.route("/download_screenshots/<video_id>")
def download_screenshots(video_id):
    dir_path = os.path.join(SCREENSHOT_FOLDER, video_id)
    import zipfile, io
    zip_stream = io.BytesIO()
    with zipfile.ZipFile(zip_stream, 'w') as zf:
        for root, _, files in os.walk(dir_path):
            for file in files:
                zf.write(os.path.join(root, file), arcname=file)
    zip_stream.seek(0)
    return send_file(zip_stream, mimetype='application/zip', as_attachment=True,
                     download_name=f"{video_id}_screenshots.zip")


@app.route("/cctv_stream")
def cctv_stream():
    return Response(generate_cctv_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stop_stream', methods=['POST'])
def stop_stream():
    global stream_active
    stream_active = False
    logging.info("Stream stopped.")
    return jsonify({"message": "Stream stopped"})



@app.route("/stream")
def stream_page():
    return render_template("stream.html")

def append_to_history(metadata):
    try:
        # Load existing history or start new if the history file is empty or doesn't exist
        if os.path.exists(HISTORY_FILE) and os.path.getsize(HISTORY_FILE) > 0:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        else:
            history = []

        # Append the new metadata
        history.append(metadata)

        # Save the updated history back to the file
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=4)

        print("✅ Metadata appended to history.")
    except Exception as e:
        print(f"❌ Failed to save metadata: {e}")

# Simulated real-time processing endpoint
import time

@app.route("/process_realtime_stream", methods=["POST"])
def process_realtime_stream():
    try:
        # Simulate stream processing logic
        session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        video_filename = f"realtime_stream_{session_id}.mp4v"
        processed_video_path = os.path.join(PROCESSED_FOLDER, video_filename)

        # Simulate creating a black video (for testing purposes)
        height, width = 480, 640
        fps = 10  # Assume real-time footage is 30 FPS
        out = cv2.VideoWriter(processed_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

        # List to store detected anomalies
        detected_behaviors = []

        for frame_count in range(30):  # Simulate 30 frames (3 seconds)
            frame = 255 * np.ones((height, width, 3), dtype=np.uint8)  # Dummy frame for testing
            label, confidence = predict_behavior(frame)

            # If an anomaly is detected, add to the list
            if label != "normal" and label != "error":
                detected_behaviors.append(label)

                # Save anomaly screenshots
                screenshot_folder = os.path.join(SCREENSHOT_FOLDER, session_id)
                os.makedirs(screenshot_folder, exist_ok=True)
                screenshot_filename = f"frame_{frame_count}_{label.replace(' ', '')}_{confidence:.2f}.jpg"
                screenshot_path = os.path.join(screenshot_folder, screenshot_filename)
                cv2.imwrite(screenshot_path, frame)

            # Write frame to the output video
            out.write(frame)

            # Sleep to match real-time speed
            time.sleep(1 / fps)  # Wait for the correct amount of time between frames

        out.release()

        # Prepare metadata
        metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "video_name": video_filename,
            "output_video": processed_video_path,
            "screenshot_paths": [f"{session_id}/{screenshot_filename}" for screenshot_filename in
                                 os.listdir(screenshot_folder)],
            "anomalies": list(set(detected_behaviors))  # Include actual anomalies detected
        }

        # Save metadata to history file
        append_to_history(metadata)

        return jsonify(metadata)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# Fetch the most recent processed metadata (optional for frontend)
@app.route("/latest_metadata", methods=["GET"])
def get_latest_metadata():
    try:
        # Load the most recent metadata from the history file
        with open("history1.json", "r") as f:
            history = json.load(f)

        # Get the most recent metadata (last entry in the history list)
        latest_metadata = history[-1] if history else {}

        return jsonify(latest_metadata)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
