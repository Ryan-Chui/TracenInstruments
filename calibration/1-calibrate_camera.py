import os
import argparse
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scanner.acquisition import Camera
from scanner.calibration import CameraCalibrator

def parse_args():
    parser = argparse.ArgumentParser(description='Calibrate an ELP or other USB camera with ChArUco images.')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0).')
    parser.add_argument('--width', type=int, default=1920, help='Requested capture width.')
    parser.add_argument('--height', type=int, default=1080, help='Requested capture height.')
    parser.add_argument('--fps', type=int, default=20, help='Requested capture rate.')
    parser.add_argument('--images', default='./data/CalibrationImgs/elp_camera/',
                        help='Folder for captured calibration images.')
    parser.add_argument('--results', default='./data/calib_results/elp_camera/',
                        help='Folder for camera calibration output.')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.images, exist_ok=True)

    cam_mtx = None
    cam_dist = None
    if os.path.exists(os.path.join(args.results, 'cam_mtx.npy')):
        cam_mtx = np.load(os.path.join(args.results, 'cam_mtx.npy'))
    if os.path.exists(os.path.join(args.results, 'cam_dist.npy')):
        cam_dist = np.load(os.path.join(args.results, 'cam_dist.npy'))

    remove_dist = False
    # Start camera and calibrator
    cam = Camera(args.camera, width=args.width, height=args.height, fps=args.fps)
    calibrator = CameraCalibrator(args.width, args.height, cam_mtx, cam_dist)

    while True:
        frame = cam.get_frame()

        if frame is not None:
            # Reset parameters
            og_frame = frame.copy()

            # Convert to gray
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers and corners
            ret, frame, _, _ = calibrator.detect_markers(frame, gray)

            if remove_dist:
                # Undistort image
                frame = cam.remove_dist(frame)
            
            resized_frame = cv2.resize(frame, (960, 540))
            cv2.imshow('frame', resized_frame)

        keyPressed = cv2.waitKey(1)
        if keyPressed == ord('q'):
            cam.stop_cam()
            break
        if keyPressed == ord('d'):
            remove_dist = not remove_dist
        if keyPressed == ord('c'):
            if ret is not None:
                n = 0
                while os.path.exists(os.path.join(args.images, f'calibrate_{n}.jpg')):
                    n += 1
                cv2.imwrite(os.path.join(args.images, f'calibrate_{n}.jpg'), og_frame)
        if keyPressed == ord('k'):
            cam_mtx, cam_dist = calibrator.calibrate(args.images)
            cam.mtx = cam_mtx
            cam.dist = cam_dist

            # Save calibration results
            np.save(os.path.join(args.results, 'cam_mtx.npy'), cam_mtx)
            np.save(os.path.join(args.results, 'cam_dist.npy'), cam_dist)