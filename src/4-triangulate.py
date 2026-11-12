import glob
import json
import os
import cv2
import numpy as np

from scanner.triangulation import Triangulate
from scanner.calibration.projcam_calibrator import calibration_projector_size
from scanner.utils import visualize

# Settings
# Recording made by 3-capture_decode.py: 'auto' = the newest, or e.g. './data/recordings/record_3/'
recording = 'auto'

# Calibration results from parts 1 and 2 ('auto' = newest camera calibration)
cam_result_folder = 'auto'
proj_result_folder = './data/calib_results/my_proj/'
stereo_result_folder = './data/calib_results/my_stereo/'
# Part 2's photo folder, used to look up the projector resolution during calibration
calib_image_folder = './data/CalibrationImgs/my_projector/'

# Keep points this far from the camera (meters)
depth_range_m = (0.2, 3.0)

output_root = './data/point_clouds/'



def newest(pattern):
    paths = glob.glob(pattern)
    return max(paths, key=os.path.getmtime) if paths else None


def plane_fit_report(pts):
    """Fit a plane and report how far the points are from it (for the flat-wall test)."""
    p = pts.T if pts.shape[0] == 3 else pts
    for _ in range(3): # refit without the worst points
        center = p.mean(axis=0)
        _, _, vt = np.linalg.svd(p - center, full_matrices=False)
        normal = vt[2]
        dist = (p - center) @ normal
        keep = np.abs(dist) < max(3 * dist.std(), 1e-4)
        p = p[keep]
    dist = (p - p.mean(axis=0)) @ normal
    print(f'Best-fit plane: {np.sqrt(np.mean(dist ** 2)) * 1000:.2f} mm RMS from the plane '
          f'({len(p)} points). If you scanned a flat wall, this is the scan noise.')


if __name__ == '__main__':
    # Recording
    if recording == 'auto':
        recording = newest('./data/recordings/*/h_pixels.npy')
        if recording is None:
            raise SystemExit('No recordings found. Run 3-capture_decode.py first.')
        recording = os.path.dirname(recording)
    print(f'Recording: {recording}')
    h_pixels = np.load(os.path.join(recording, 'h_pixels.npy'))
    v_pixels = np.load(os.path.join(recording, 'v_pixels.npy'))
    img_white = cv2.cvtColor(cv2.imread(os.path.join(recording, 'frame_1.jpg'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    cam_w, cam_h = img_white.shape[1], img_white.shape[0]

    scan_info = {}
    if os.path.exists(os.path.join(recording, 'scan.json')):
        with open(os.path.join(recording, 'scan.json')) as fp:
            scan_info = json.load(fp)

    # Calibration
    if cam_result_folder == 'auto':
        cam_result_folder = os.path.dirname(newest('./data/calib_results/*/cam_mtx.npy'))
    cam_mtx = np.load(os.path.join(cam_result_folder, 'cam_mtx.npy'))
    cam_dist = np.load(os.path.join(cam_result_folder, 'cam_dist.npy'))
    proj_mtx = np.load(os.path.join(proj_result_folder, 'proj_mtx.npy'))
    proj_dist = np.load(os.path.join(proj_result_folder, 'proj_dist.npy'))
    proj_R = np.load(os.path.join(stereo_result_folder, 'R.npy'))
    proj_T = np.load(os.path.join(stereo_result_folder, 'T.npy'))
    print(f'Camera calibration: {cam_result_folder}   projector: {proj_result_folder}   stereo: {stereo_result_folder}')
    if abs(cam_mtx[0, 2] - cam_w / 2) > cam_w * 0.1 or abs(cam_mtx[1, 2] - cam_h / 2) > cam_h * 0.1:
        print(f'WARNING: the camera calibration does not look like it was made at {cam_w}x{cam_h}.')

    # Projector resolution during the scan and during calibration
    calib = calibration_projector_size(calib_image_folder)
    if calib is None:
        raise SystemExit(f'Could not find the projector resolution used for calibration in {calib_image_folder}. '
                         'Run 2b-recalibrate.py once (it records it), or check calib_image_folder.')
    calib_proj_w, calib_proj_h = calib[0], calib[1]
    proj_w = scan_info.get('proj_width', calib_proj_w)
    proj_h = scan_info.get('proj_height', calib_proj_h)
    print(f'Projector: {proj_w}x{proj_h} during the scan, {calib_proj_w}x{calib_proj_h} during calibration')
    if (proj_w, proj_h) != (calib_proj_w, calib_proj_h):
        print('WARNING: the projector resolution changed since calibration. The calibration is rescaled, '
              'but redoing part 2 at the new resolution is more accurate.')

    # Triangulate
    triangulator = Triangulate(h_pixels, v_pixels, (cam_w, cam_h), cam_mtx, cam_dist,
                               (proj_w, proj_h), (calib_proj_w, calib_proj_h), proj_mtx, proj_dist,
                               proj_R, proj_T, recording)
    cam_pts, proj_pts, colors = triangulator.get_cam_proj_pts(img_white)
    if len(cam_pts) == 0:
        raise SystemExit('No decoded pixels in this recording. Look at its decoded.png from part 3.')
    pts_3d = triangulator.triangulate(cam_pts, proj_pts)
    pts_3d, colors = triangulator.filter_3d_pts(pts_3d, colors, z_range=depth_range_m)
    if pts_3d.shape[1] == 0:
        raise SystemExit(f'No points between {depth_range_m[0]} and {depth_range_m[1]} m. Check depth_range_m.')

    z = pts_3d[2]
    print(f'{pts_3d.shape[1]} points, depth {np.percentile(z, 5):.3f} to {np.percentile(z, 95):.3f} m '
          f'(median {np.median(z):.3f} m)')
    plane_fit_report(pts_3d)

    # Save and show
    output_folder = os.path.join(output_root, os.path.basename(os.path.normpath(recording)))
    os.makedirs(output_folder, exist_ok=True)
    np.save(os.path.join(output_folder, 'pts_3d.npy'), pts_3d)
    np.save(os.path.join(output_folder, 'colors.npy'), colors)
    visualize.plot_point_cloud(pts_3d, colors, output_folder)
