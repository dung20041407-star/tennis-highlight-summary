from flask import Flask, render_template, request, jsonify
import os
import shutil
import subprocess
import json

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")
RALLY_FOLDER = os.path.join(BASE_DIR, "static", "rallies")

INPUT_VIDEO_FOLDER = os.path.join(BASE_DIR, "input_video")
OUTPUT_VIDEO_FOLDER = os.path.join(BASE_DIR, "output_video")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(RALLY_FOLDER, exist_ok=True)
os.makedirs(INPUT_VIDEO_FOLDER, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload_video():

    if "video" not in request.files:
        return jsonify({"success": False})

    video = request.files["video"]

    filename = video.filename

    save_path = os.path.join(UPLOAD_FOLDER, filename)

    video.save(save_path)

    return jsonify({
        "success": True,
        "filename": filename
    })

@app.route("/run", methods=["POST"])
def run_pipeline():

    data = request.get_json()

    filename = data["filename"]

    upload_path = os.path.join(UPLOAD_FOLDER, filename)

    for f in os.listdir(INPUT_VIDEO_FOLDER):
        os.remove(os.path.join(INPUT_VIDEO_FOLDER, f))

    shutil.copy(upload_path, INPUT_VIDEO_FOLDER)

    try:
        subprocess.run(
            ["python", "main.py"],
            cwd=BASE_DIR,
            check=True
        )

    except subprocess.CalledProcessError:
        return jsonify({
            "success": False,
            "message": "Pipeline failed."
        })

    result_video = None

    for file in os.listdir(OUTPUT_VIDEO_FOLDER):

        if file.endswith("_result.mp4"):

            shutil.copy(
                os.path.join(OUTPUT_VIDEO_FOLDER, file),
                os.path.join(OUTPUT_FOLDER, file)
            )

            result_video = file
            break

    rally_files = []

    rally_src = os.path.join(OUTPUT_VIDEO_FOLDER, "rallies")

    if os.path.exists(rally_src):

        for file in os.listdir(rally_src):

            shutil.copy(
                os.path.join(rally_src, file),
                os.path.join(RALLY_FOLDER, file)
            )

            rally_files.append(file)

    bounce_count = 0

    event_path = os.path.join(OUTPUT_VIDEO_FOLDER, "events.json")

    if os.path.exists(event_path):

        with open(event_path, "r") as f:
            bounce_count = len(json.load(f)["events"])

    return jsonify({
        "success": True,
        "video": result_video,
        "bounce": bounce_count,
        "rallies": rally_files
    })


if __name__ == "__main__":
    app.run(debug=True)