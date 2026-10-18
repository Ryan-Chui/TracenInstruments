"""
Redo the projector-camera calibration from photos already saved by 2-calibrate_projcam.py.
No projector or camera needed.
"""
import glob
import json
import os
import numpy as np

from scanner.calibration import ProjCamCalibrator
from scanner.calibration.projcam_calibrator import infer_projector_size

# ---------------- Your settings ----------------
# Same folders as 2-calibrate_projcam.py
image_folder = './data/CalibrationImgs/my_projector/'
proj_result_folder = './data/calib_results/my_proj/'
result_folder = './data/calib_results/my_stereo/'

# 'auto' = the folder holding your newest camera calibration from part 1
cam_result_folder = 'auto'
cam_width, cam_height = (1920, 1080)

# Projector resolution. None = read it from the capture session, or work it out from the saved dot positions.
# If neither works, set it to the "Projector resolution" that 2-calibrate_projcam.py printed, e.g. (1280, 800)
proj_size = None

# Height of the ChArUco above the white surface in mm. None = the value used when the photos were taken
board_offset_mm = None

calibrate_proj = True
circle_grid_size = (4, 11)
# -----------------------------------------------


def newest_camera_folder():
    paths = glob.glob('./data/calib_results/*/cam_mtx.npy')
    if not paths:
        raise SystemExit('No camera calibration found. Run 1-calibrate_camera.py first.')
    return os.path.dirname(max(paths, key=os.path.getmtime))


if __name__ == '__main__':
    if not glob.glob(os.path.join(image_folder, '*.jpg')):
        raise SystemExit(f'No photos in {image_folder}')

    # Projector settings from the capture session
    session = {}
    session_file = os.path.join(image_folder, 'projector.json')
    if os.path.exists(session_file):
        with open(session_file) as fp:
            session = json.load(fp)

    circle_r = 15
    if proj_size is not None:
        proj_width, proj_height = proj_size
        source = 'your setting'
    elif session:
        proj_width, proj_height, circle_r = session['width'], session['height'], session['circle_r']
        source = 'the capture session'
    else:
        guess = infer_projector_size(image_folder)
        if guess is None:
            raise SystemExit('Could not work out the projector resolution from the photos. Set proj_size at the top of '
                             'this script to the "Projector resolution" that 2-calibrate_projcam.py printed.')
        proj_width, proj_height, circle_r = guess
        source = 'the saved dot positions'
        # Record it, so later steps don't have to work it out again
        with open(session_file, 'w') as fp:
            json.dump(dict(width=proj_width, height=proj_height, circle_r=circle_r,
                           board_offset_mm=board_offset_mm if board_offset_mm is not None else 0.0), fp)
    if board_offset_mm is None:
        board_offset_mm = session.get('board_offset_mm', 0.0)

    if cam_result_folder == 'auto':
        cam_result_folder = newest_camera_folder()
    cam_mtx = np.load(os.path.join(cam_result_folder, 'cam_mtx.npy'))
    cam_dist = np.load(os.path.join(cam_result_folder, 'cam_dist.npy'))

    print(f'Camera calibration: {cam_result_folder}  (fx = {cam_mtx[0,0]:.0f}, center = {cam_mtx[0,2]:.0f}, {cam_mtx[1,2]:.0f})')
    print(f'Projector resolution: {proj_width}x{proj_height} (from {source})')
    print(f'ChArUco height above the white surface: {board_offset_mm} mm')

    proj_mtx, proj_dist = None, None
    if not calibrate_proj:
        proj_mtx = np.load(os.path.join(proj_result_folder, 'proj_mtx.npy'))
        proj_dist = np.load(os.path.join(proj_result_folder, 'proj_dist.npy'))

    calibrator = ProjCamCalibrator(cam_width, cam_height, cam_mtx, cam_dist, proj_width, proj_height, proj_mtx, proj_dist,
                                   circle_grid_size, circle_r, board_offset_m=board_offset_mm / 1000.0)
    proj_mtx, proj_dist, proj_R, proj_T, R1, R2, P1, P2, Q = calibrator.calibrate(image_folder, calibrate_proj)

    os.makedirs(proj_result_folder, exist_ok=True)
    os.makedirs(result_folder, exist_ok=True)
    if calibrate_proj:
        np.save(os.path.join(proj_result_folder, 'proj_mtx.npy'), proj_mtx)
        np.save(os.path.join(proj_result_folder, 'proj_dist.npy'), proj_dist)
    for name, value in dict(R=proj_R, T=proj_T, R1=R1, R2=R2, P1=P1, P2=P2).items():
        np.save(os.path.join(result_folder, f'{name}.npy'), value)
    print(f'\nSaved projector results to {proj_result_folder} and stereo results to {result_folder}')
