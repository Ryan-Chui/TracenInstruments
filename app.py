from flask import Flask, Response, render_template, jsonify
import cv2
import face_alignment as alignment

app = Flask(__name__)
CAMERA_INDEX = 0

def generate_frames():
    """Capture, process, and stream webcam frames."""
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        return

    try:
        face_cascade = alignment.create_face_detector()

        while True:
            success, frame = camera.read()
            if success:
                frame = cv2.flip(frame, 1)

            if not success:
                break

            frame_height, frame_width = frame.shape[:2]
            target = alignment.target_box(frame_width, frame_height)

            target_left, target_top, target_right, target_bottom = target

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(80, 80),
            )

            # Draw the target box.
            cv2.rectangle(
                frame,
                (target_left, target_top),
                (target_right, target_bottom),
                (255, 200, 0),
                2,
            )

            if len(faces) == 0:
                alignment.draw_label(
                    frame,
                    "place your face in the box",
                    (0, 165, 255),
                    (20, 40),
                )
            else:
                face_x, face_y, face_width, face_height = max(
                    faces, key=lambda face: face[2] * face[3]
                )

                face_box = (
                    face_x,
                    face_y,
                    face_x + face_width,
                    face_y + face_height,
                )

                message, color = alignment.alignment_message(
                    face_box, target
                )

                cv2.rectangle(
                    frame,
                    (face_box[0], face_box[1]),
                    (face_box[2], face_box[3]),
                    color,
                    2,
                )

                alignment.draw_label(
                    frame, message, color, (20, 40)
                )

            # Convert the processed frame to JPEG.
            encoded, buffer = cv2.imencode(".jpg", frame)

            if not encoded:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )

    finally:
        camera.release()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

@app.route("/health")
def health():
    camera = None

    try:
        detector = alignment.create_face_detector()

        if detector.empty():
            raise RuntimeError("Face detector could not be loaded.")

        camera = cv2.VideoCapture(CAMERA_INDEX)

        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open camera index {CAMERA_INDEX}."
            )

        success, frame = camera.read()

        if not success or frame is None:
            raise RuntimeError("Could not read a camera frame.")

        return jsonify({
            "ok": True,
            "message": "Camera and face detector are ready."
        })

    except Exception as error:
        app.logger.exception("Health check failed")

        return jsonify({
            "ok": False,
            "error": str(error)
        }), 500

    finally:
        if camera is not None:
            camera.release()


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
    )