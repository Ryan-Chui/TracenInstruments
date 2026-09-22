import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

import cv2


WINDOW_NAME = "Face alignment"
TARGET_WIDTH_RATIO = 0.34
TARGET_HEIGHT_RATIO = 0.46
SIZE_TOLERANCE = 0.12
CENTER_TOLERANCE = 0.10
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"
CASCADE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/"
    "haarcascade_frontalface_default.xml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Show a camera feed and guide a face into a centered target box."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera index to open (default: 0).",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=80,
        help="Smallest face dimension to detect in pixels (default: 80).",
    )
    return parser.parse_args()


def target_box(frame_width, frame_height):
    target_width = int(frame_width * TARGET_WIDTH_RATIO)
    target_height = int(frame_height * TARGET_HEIGHT_RATIO)
    left = (frame_width - target_width) // 2
    top = (frame_height - target_height) // 2
    return left, top, left + target_width, top + target_height


def alignment_message(face_box, target):
    face_left, face_top, face_right, face_bottom = face_box
    target_left, target_top, target_right, target_bottom = target

    face_width = face_right - face_left
    face_height = face_bottom - face_top
    target_width = target_right - target_left
    target_height = target_bottom - target_top

    width_ratio = face_width / target_width
    height_ratio = face_height / target_height

    if width_ratio < 1 - SIZE_TOLERANCE or height_ratio < 1 - SIZE_TOLERANCE:
        return "move closer to the camera", (0, 165, 255)
    if width_ratio > 1 + SIZE_TOLERANCE or height_ratio > 1 + SIZE_TOLERANCE:
        return "move farther from the camera", (0, 165, 255)

    face_center_x = (face_left + face_right) / 2
    face_center_y = (face_top + face_bottom) / 2
    target_center_x = (target_left + target_right) / 2
    target_center_y = (target_top + target_bottom) / 2

    horizontal_error = (face_center_x - target_center_x) / target_width
    vertical_error = (face_center_y - target_center_y) / target_height

    if horizontal_error < -CENTER_TOLERANCE:
        return "move right", (0, 165, 255)
    if horizontal_error > CENTER_TOLERANCE:
        return "move left", (0, 165, 255)
    if vertical_error < -CENTER_TOLERANCE:
        return "move down", (0, 165, 255)
    if vertical_error > CENTER_TOLERANCE:
        return "move up", (0, 165, 255)

    return "aligned", (80, 220, 80)


def draw_label(frame, text, color, position):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def create_face_detector():
    cascade_path = Path(__file__).with_name("models") / CASCADE_FILENAME
    if not cascade_path.exists():
        cascade_path.parent.mkdir(exist_ok=True)
        try:
            with urlopen(CASCADE_URL, timeout=15) as response:
                cascade_path.write_bytes(response.read())
        except Exception as error:
            raise RuntimeError(
                "The face detector model is missing and could not be downloaded. "
                f"Download {CASCADE_URL} to {cascade_path}."
            ) from error

    xobjdetect = getattr(cv2, "xobjdetect", None)
    if xobjdetect is not None and hasattr(xobjdetect, "CascadeClassifier"):
        detector = xobjdetect.CascadeClassifier(str(cascade_path))
    elif hasattr(cv2, "CascadeClassifier"):
        detector = cv2.CascadeClassifier(str(cascade_path))
    else:
        raise RuntimeError(
            "This OpenCV build does not provide a cascade classifier. "
            "Reinstall with: python -m pip install --force-reinstall "
            '"opencv-contrib-python>=5"'
        )

    if detector.empty():
        raise RuntimeError(f"Could not load the face detector model: {cascade_path}")
    return detector


def run(camera_index, min_face_size):
    face_cascade = create_face_detector()
    if face_cascade.empty():
        raise RuntimeError("Could not load OpenCV's face detector.")

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print("Could not read a frame from the camera.", file=sys.stderr)
                break

            frame_height, frame_width = frame.shape[:2]
            target = target_box(frame_width, frame_height)
            target_left, target_top, target_right, target_bottom = target

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(min_face_size, min_face_size),
            )

            cv2.rectangle(
                frame,
                (target_left, target_top),
                (target_right, target_bottom),
                (255, 200, 0),
                2,
            )

            if len(faces) == 0:
                draw_label(frame, "place your face in the box", (0, 165, 255), (20, 40))
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
                message, color = alignment_message(face_box, target)
                cv2.rectangle(
                    frame,
                    (face_box[0], face_box[1]),
                    (face_box[2], face_box[3]),
                    color,
                    2,
                )
                draw_label(frame, message, color, (20, 40))

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    arguments = parse_args()
    try:
        run(arguments.camera, arguments.min_face_size)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)