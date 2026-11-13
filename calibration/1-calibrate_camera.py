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
    parser.add_argument('--exposure', type=float, default=None,
                        help='Manual exposure value; lower values darken the image.')
    parser.add_argument('--images', default='./data/CalibrationImgs/elp_camera/',
                        help='Folder for captured calibration images.')
    parser.add_argument('--results', default='./data/calib_results/elp_camera/',
                        help='Folder for camera calibration output.')
    parser.add_argument('--reuse-existing', action='store_true',
                        help='Use existing intrinsics as the calibration initial guess.')
    parser.add_argument('--min-images', type=int, default=10,
                        help='Minimum captured images required before calibration.')
    return parser.parse_args()


def run(args):
    os.makedirs(args.results, exist_ok=True)
    os.makedirs(args.images, exist_ok=True)

    cam_mtx = None
    cam_dist = None
    if args.reuse_existing:
        if os.path.exists(os.path.join(args.results, 'cam_mtx.npy')):
            cam_mtx = np.load(os.path.join(args.results, 'cam_mtx.npy'))
        if os.path.exists(os.path.join(args.results, 'cam_dist.npy')):
            cam_dist = np.load(os.path.join(args.results, 'cam_dist.npy'))

    remove_dist = False
    # Start camera and calibrator
    cam = None
    try:
        cam = Camera(
            args.camera,
            width=args.width,
            height=args.height,
            fps=args.fps,
            exposure=args.exposure,
        )
        calibrator = CameraCalibrator(args.width, args.height, cam_mtx, cam_dist)
        ret = False
        og_frame = None

        while True:
            frame = cam.get_frame()

            if frame is not None:
                og_frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, frame, _, _ = calibrator.detect_markers(frame, gray)

                if remove_dist:
                    frame = cam.remove_dist(frame)

                resized_frame = cv2.resize(frame, (960, 540))
                cv2.imshow('frame', resized_frame)

            keyPressed = cv2.waitKey(1)
            if keyPressed == ord('q'):
                break
            if keyPressed == ord('d'):
                remove_dist = not remove_dist
            if keyPressed == ord('c') and og_frame is not None:
                n = 0
                while os.path.exists(os.path.join(args.images, f'calibrate_{n}.jpg')):
                    n += 1
                cv2.imwrite(os.path.join(args.images, f'calibrate_{n}.jpg'), og_frame)
            if keyPressed == ord('k'):
                image_count = len(
                    [
                        name for name in os.listdir(args.images)
                        if name.lower().endswith('.jpg')
                    ]
                )
                if image_count < args.min_images:
                    print(
                        f'Calibration cancelled: only {image_count} images found; '
                        f'capture at least {args.min_images} varied images first.'
                    )
                    continue
                cam_mtx, cam_dist = calibrator.calibrate(args.images)
                cam.mtx = cam_mtx
                cam.dist = cam_dist
                np.save(os.path.join(args.results, 'cam_mtx.npy'), cam_mtx)
                np.save(os.path.join(args.results, 'cam_dist.npy'), cam_dist)
    finally:
        if cam is not None:
            cam.stop_cam()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        run(parse_args())
    except (OSError, RuntimeError, ValueError, cv2.error) as error:
        print(f'Calibration failed: {error}', file=sys.stderr)
        sys.exit(1)