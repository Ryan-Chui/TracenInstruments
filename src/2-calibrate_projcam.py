import glob
import json
import os
import time
import cv2
import numpy as np

from scanner.acquisition import Camera, ProjectorWindow
from scanner.acquisition.projector import place_window
from scanner.calibration import ProjCamCalibrator

# SETTINGS
cam_src = 0
cam_width, cam_height = (1920, 1080)
cam_fps = 30

# 'auto' = the folder holding camera calibration from part 1.
cam_result_folder = 'auto'

# True: calibrate the projector lens and the camera-projector position.
# False: reuse a projector lens calibration saved earlier and only redo the position.
calibrate_proj = True

image_folder = './data/CalibrationImgs/my_projector/'
proj_result_folder = './data/calib_results/my_proj/'
result_folder = './data/calib_results/my_stereo/'

# Projector display: None = the display that isn't the built-in screen.
# If the pattern lands on the wrong display, set the number printed next to the projector, e.g. 1
proj_screen_index = None

# Height of the printed ChArUco above the white surface the dots land on, in millimeters.
# (paper thickness doesn't matter).
board_offset_mm = 0.0

# Circle grid pattern (4 circles per row, 11 rows, asymmetric)
circle_grid_size = (4, 11)

# Automatic capture: the board must be still for STILL_FRAMES frames (average movement under STILL_PX pixels),
# and the circles must have moved at least MOVE_PX pixels since the last saved photo
STILL_FRAMES = 4
STILL_PX = 2.0
MOVE_PX = 20.0
SETTLE_S = 0.6 # wait after the projected grid moves, so the camera shows the new position


def newest_camera_folder():
    paths = glob.glob('./data/calib_results/*/cam_mtx.npy')
    if not paths:
        raise SystemExit('No camera calibration found. Run 1-calibrate_camera.py first.')
    return os.path.dirname(max(paths, key=os.path.getmtime))


def next_image_path(folder):
    n = 0
    while os.path.exists(os.path.join(folder, f'calibrate_{n}.jpg')):
        n += 1
    return os.path.join(folder, f'calibrate_{n}.jpg')


if __name__ == '__main__':
    for folder in (image_folder, proj_result_folder, result_folder):
        os.makedirs(folder, exist_ok=True)

    # Camera calibration from part 1
    if cam_result_folder == 'auto':
        cam_result_folder = newest_camera_folder()
    cam_mtx = np.load(os.path.join(cam_result_folder, 'cam_mtx.npy'))
    cam_dist = np.load(os.path.join(cam_result_folder, 'cam_dist.npy'))
    print(f'Camera calibration: {cam_result_folder}  (fx = {cam_mtx[0,0]:.0f}, center = {cam_mtx[0,2]:.0f}, {cam_mtx[1,2]:.0f})')
    print(f'ChArUco height above the white surface: {board_offset_mm} mm')
    if not (0 < cam_mtx[0, 2] < cam_width and 0 < cam_mtx[1, 2] < cam_height):
        raise SystemExit('That camera calibration does not match a '
                         f'{cam_width}x{cam_height} camera. Set cam_result_folder to your part 1 results.')

    # Window covering the projector display; its size is the projector's real resolution
    projector = ProjectorWindow('Projector', screen_index=proj_screen_index)
    proj_width, proj_height = projector.width, projector.height
    print(f'Projector resolution: {proj_width}x{proj_height}')

    # Camera window on your own screen, with keyboard focus
    cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('Camera', np.zeros((540, 960, 3), dtype=np.uint8))
    cv2.waitKey(50)
    place_window('Camera', projector.control_display)

    # Saved projector lens calibration, only when it is reused
    proj_mtx, proj_dist = None, None
    if not calibrate_proj:
        proj_mtx = np.load(os.path.join(proj_result_folder, 'proj_mtx.npy'))
        proj_dist = np.load(os.path.join(proj_result_folder, 'proj_dist.npy'))

    # Circle size scales with the projector resolution (15 px radius at 1080p)
    circle_r = max(6, int(round(15 * proj_height / 1080)))

    # Remember this session's projector settings, so 2b-recalibrate.py can redo the math without the projector
    with open(os.path.join(image_folder, 'projector.json'), 'w') as fp:
        json.dump(dict(width=proj_width, height=proj_height, circle_r=circle_r, board_offset_mm=board_offset_mm), fp)

    # Start camera and calibrator
    cam = Camera(cam_src, width=cam_width, height=cam_height, fps=cam_fps)
    calibrator = ProjCamCalibrator(cam_width, cam_height, cam_mtx, cam_dist, proj_width, proj_height, proj_mtx, proj_dist, circle_grid_size, circle_r,
                                   board_offset_m=board_offset_mm / 1000.0)

    # The grid visits 9 positions across the projector image, moving on after each saved photo
    positions = calibrator.grid_positions(cols=3, rows=3)
    pos_id = 0

    def show_grid():
        global pattern_time
        projector.show(calibrator.get_circle_grid_image(*positions[pos_id]))
        pattern_time = time.time()

    show_grid()
    n_saved = len(glob.glob(os.path.join(image_folder, '*.jpg')))
    prev_circles, last_saved, still = None, None, 0
    circles_2d_cam = None

    def save_photo(og_frame, circles):
        global n_saved, last_saved, pos_id
        im_name = next_image_path(image_folder)
        cv2.imwrite(im_name, og_frame)
        # Remember where the grid was in the projector image for this photo
        np.save(ProjCamCalibrator.proj_points_path(im_name), calibrator.circle_2d.astype(np.float32))
        n_saved += 1
        last_saved = circles.copy()
        print(f'Saved {im_name}  (grid position {pos_id + 1}/{len(positions)})')
        pos_id = (pos_id + 1) % len(positions)
        show_grid()

    print('\nPut the white area of the board under the projected dots and hold still. '
          'Photos are taken automatically.\nKeys (click the camera window first): '
          'c = save photo now, n = move the grid, k = calibrate, q = quit')

    while True:
        frame = cam.get_frame()

        if frame is not None:
            og_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect the board, then the projected circles on the same plane
            circles_2d_cam = None
            ret, frame, H = calibrator.detect_markers(frame, gray)
            if ret:
                ok, frame, circles, _ = calibrator.detect_circle_grid(frame, gray, H)
                if ok:
                    circles_2d_cam = circles

            # Is the board holding still?
            if circles_2d_cam is not None and prev_circles is not None:
                motion = np.linalg.norm(circles_2d_cam - prev_circles, axis=2).mean()
                still = still + 1 if motion < STILL_PX else 0
            else:
                still = 0
            prev_circles = circles_2d_cam

            # Save automatically when still, settled, and in a new spot
            if circles_2d_cam is not None and still >= STILL_FRAMES and time.time() - pattern_time > SETTLE_S:
                moved = last_saved is None or np.linalg.norm(circles_2d_cam - last_saved, axis=2).mean() > MOVE_PX
                if moved:
                    save_photo(og_frame, circles_2d_cam)
                    prev_circles, still = None, 0

            # Show camera frame with status
            resized_frame = cv2.resize(frame, (960, 540))
            if not ret:
                status = 'Board not found'
            elif circles_2d_cam is None:
                status = 'Board found - move the white area under the dots'
            else:
                status = 'Hold still...'
            cv2.putText(resized_frame, f'{status}   saved: {n_saved}   grid {pos_id + 1}/{len(positions)}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Camera', resized_frame)

        keyPressed = cv2.waitKey(1)
        if keyPressed == ord('q'):
            cam.stop_cam()
            break
        if keyPressed == ord('n'):
            pos_id = (pos_id + 1) % len(positions)
            show_grid()
            prev_circles, still = None, 0
        if keyPressed == ord('c'):
            if circles_2d_cam is not None and time.time() - pattern_time > SETTLE_S:
                save_photo(og_frame, circles_2d_cam)
                prev_circles, still = None, 0
        if keyPressed == ord('k'):
            # Calibrate projector and stereo configuration
            try:
                proj_mtx, proj_dist, proj_R, proj_T, R1, R2, P1, P2, Q = calibrator.calibrate(image_folder, calibrate_proj)

                if calibrate_proj:
                    # Save calibration results
                    np.save(os.path.join(proj_result_folder, 'proj_mtx.npy'), proj_mtx)
                    np.save(os.path.join(proj_result_folder, 'proj_dist.npy'), proj_dist)

                # Save stereo calibration results
                np.save(os.path.join(result_folder, 'R.npy'), proj_R)
                np.save(os.path.join(result_folder, 'T.npy'), proj_T)
                np.save(os.path.join(result_folder, 'R1.npy'), R1)
                np.save(os.path.join(result_folder, 'R2.npy'), R2)
                np.save(os.path.join(result_folder, 'P1.npy'), P1)
                np.save(os.path.join(result_folder, 'P2.npy'), P2)
                print(f'\nSaved projector results to {proj_result_folder} and stereo results to {result_folder}')
            except Exception as e:
                print('Calibration failed!', e)
