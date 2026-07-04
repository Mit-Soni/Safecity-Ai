import os
import io
import time
import torch
import cv2
from werkzeug.utils import secure_filename
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory, render_template_string

app = Flask(__name__)

# Configuration paths
MODEL_WEIGHTS_PATH = "runs/train/ex6_fine_tune/weights/best.pt"
UPLOAD_FOLDER = r"D:\training porject safecity\app_uploads"
OUTPUT_FOLDER = r"D:\training porject safecity\runs\detect"

# Normalize paths for clean Windows OS handling
UPLOAD_FOLDER = os.path.abspath(os.path.normpath(UPLOAD_FOLDER))
OUTPUT_FOLDER = os.path.abspath(os.path.normpath(OUTPUT_FOLDER))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

print("Loading custom YOLOv5 model...")
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_WEIGHTS_PATH)
model.conf = 0.40  
print("Model loaded successfully!")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SafeCityAI Traffic Hub</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; background-color: #f4f4f9; color: #333; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.05); }
        h1 { text-align: center; color: #1e3a8a; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0; }
        .form-group { padding: 20px; border: 2px dashed #cbd5e1; border-radius: 8px; background: #fafafa; text-align: center; }
        h3 { margin-top: 0; color: #2c3e50; }
        button { background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; margin-top: 10px; width: 100%; font-weight: bold; }
        button:hover { background: #2980b9; }
        .result-box { margin-top: 25px; padding: 20px; border: 1px solid #2ecc71; border-radius: 6px; background: #eafaf1; text-align: center; }
        img, video { max-width: 100%; border-radius: 6px; margin-top: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); background-color: #000; }
    </style>
</head>
<body>
    <div class="container">
        <h1>SafeCityAI Traffic Enforcement Panel</h1>
        
        <div class="grid">
            <div class="form-group">
                <h3>📷 Upload Traffic Image</h3>
                <form action="/process_image" method="POST" enctype="multipart/form-data">
                    <input type="file" name="image" accept="image/*" required>
                    <button type="submit">Analyze Image</button>
                </form>
            </div>

            <div class="form-group">
                <h3>🎥 Upload Traffic Video</h3>
                <form action="/process_video" method="POST" enctype="multipart/form-data">
                    <input type="file" name="video" accept="video/*" required>
                    <button type="submit">Analyze Video</button>
                </form>
            </div>
        </div>

        {% if image_url %}
        <div class="result-box">
            <h3>Processed Image Detections:</h3>
            <img src="{{ image_url }}" alt="Processed Frame">
        </div>
        {% endif %}

        {% if video_url %}
        <div class="result-box">
            <h3>Processed Video Feed (Native HTML5 Playback):</h3>
            <video controls autoplay muted playsinline>
                <source src="{{ video_url }}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(DASHBOARD_HTML, video_url=None, image_url=None)

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route("/process_image", methods=["POST"])
def process_image():
    if "image" not in request.files:
        return "No image file uploaded", 400
        
    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
        results = model(img)
        results.render()
        
        output_img = Image.fromarray(results.ims[0])
        secure_name = secure_filename(file.filename)
        base_name = os.path.splitext(secure_name)[0]
        
        output_filename = f"res_{base_name}_{int(time.time())}.jpg"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        output_img.save(output_path, "JPEG")
        
        runtime_url = f"/outputs/{output_filename}"
        return render_template_string(DASHBOARD_HTML, image_url=runtime_url, video_url=None)
    except Exception as e:
        return f"Image pipeline failure: {str(e)}", 500

@app.route("/process_video", methods=["POST"])
def process_video():
    if "video" not in request.files:
        return "No video file uploaded", 400

    video_file = request.files["video"]
    if video_file.filename == "":
        return "No selected file", 400

    filename = secure_filename(video_file.filename)
    raw_video_path = os.path.abspath(os.path.normpath(os.path.join(app.config['UPLOAD_FOLDER'], filename)))
    
    # Wipe old temporary file instances to prevent access locks
    if os.path.exists(raw_video_path):
        try: os.remove(raw_video_path)
        except: pass

    video_file.save(raw_video_path)

    # Multi-backend reader hook
    cap = cv2.VideoCapture(raw_video_path)
    if not cap.isOpened():
        cap = cv2.VideoCapture(raw_video_path, cv2.CAP_MSMF)
        
    if not cap.isOpened():
        return f"Error: OpenCV completely failed to read input context for '{filename}'.", 400

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    base_name = os.path.splitext(filename)[0]
    output_filename = f"web_{base_name}_{int(time.time())}.mp4"
    output_path = os.path.abspath(os.path.normpath(os.path.join(OUTPUT_FOLDER, output_filename)))

    # FIX: Explicitly forcing Windows Microsoft Media Foundation pipeline with 'avc1'
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, cv2.CAP_MSMF, fourcc, fps, (width, height))

    # Fallback Option 1: Try default pipeline if MSMF is locked
    if not out.isOpened():
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
    # Fallback Option 2: Fall back to 'mp4v' if system completely rejects H.264 compilation
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        frames_written = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model(frame_rgb)
            results.render()
            
            processed_frame = cv2.cvtColor(results.ims[0], cv2.COLOR_RGB2BGR)
            out.write(processed_frame)
            frames_written += 1

        if frames_written == 0:
            return "Processing error: Output file stream received 0 valid frames.", 500

    except Exception as e:
        return f"Video rendering stream processing failure: {str(e)}", 500
    finally:
        cap.release()
        out.release()  # Essential step: Forces Windows to append the missing 'moov' atom index header

    runtime_url = f"/outputs/{output_filename}"
    return render_template_string(DASHBOARD_HTML, video_url=runtime_url, image_url=None)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Missing 'image' file."}), 400
    file = request.files["image"]
    try:
        img_bytes = file.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        results = model(image)
        predictions = results.xyxy[0].cpu().numpy()
        output = []
        for pred in predictions:
            xmin, ymin, xmax, ymax, confidence, class_id = pred
            raw_label = model.names[int(class_id)]
            class_name = "No_Helmet" if "helmet" in raw_label.lower() else "No_Seatbelt" if "seatbelt" in raw_label.lower() else raw_label
            width = int(xmax - xmin)
            height = int(ymax - ymin)
            output.append({
                "class": class_name,
                "confidence": float(round(confidence, 2)),
                "box": [int(xmin), int(ymin), width, height]
            })
        return jsonify(output), 200
    except Exception as e:
        return jsonify({"error": f"Inference failure: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)