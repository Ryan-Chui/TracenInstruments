# Face Alignment

A small OpenCV script that uses a camera feed to guide a person's face into a centered target box.

## Setup

Create and activate a virtual environment if desired, then install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

The project requires OpenCV 5. The script uses
`cv2.xobjdetect.CascadeClassifier` when that module is available and falls back
to OpenCV 5's built-in `cv2.CascadeClassifier` for wheels that do not expose
the `xobjdetect` namespace.

## Run

```powershell
python .\face_alignment.py
```

To use another camera, pass its device index:

```powershell
python .\face_alignment.py --camera 1
```

Press `Q` while the camera window is focused to quit.

## Guidance

The camera window displays:

- A blue target box in the center of the frame
- A rectangle around the detected face
- A message to move closer or farther away when the face size is incorrect
- A message to move left, right, up, or down when the face is off-center
- `aligned` when the face is within the target size and position tolerances

On first run, the script downloads the official OpenCV Haar cascade model into
the ignored `models` folder.

If a different OpenCV version is installed, replace it with the supported version:

```powershell
python -m pip install --force-reinstall "opencv-contrib-python>=5"
```